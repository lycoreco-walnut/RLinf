# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LingBot-VA adapter for RLinf embodied rollout and SFT."""

from __future__ import annotations

import atexit
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation as R

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lingbotva.history_buffer import (
    LingbotVAEpisodeState,
    select_key_frames,
)
from rlinf.models.embodiment.lingbotva.observation_adapter import (
    LingbotVAObservationAdapter,
)
from rlinf.models.embodiment.lingbotva.utils import (
    _extend_import_path,
    _extract_transformer_state_dict,
    export_official_transformer_checkpoint,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

try:
    from wan_va.modules.model import WanTransformer3DModel
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - import path is injected by get_model.
    raise ModuleNotFoundError(
        "LingBot-VA requires cfg.lingbotva.repo_path to be inserted into sys.path "
        "before importing rlinf.models.embodiment.lingbotva.lingbotva_action_model."
    ) from exc


def _resolve_cuda_local_rank() -> int:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank)
        except ValueError as exc:
            raise ValueError(f"Invalid LOCAL_RANK value: {local_rank!r}") from exc
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def _resolve_runtime_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    repo_root = os.environ.get("REPO_PATH")
    if repo_root:
        return (Path(repo_root) / path).resolve()
    return path.resolve()


def _load_transformer_config(transformer_path: Path) -> dict[str, Any]:
    config_path = transformer_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"LingBot-VA transformer config not found at {config_path}."
        )
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    return {key: value for key, value in config_data.items() if not key.startswith("_")}


class _WanVATrainingContext:
    def __init__(self, config: Any, torch_dtype: torch.dtype) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.dtype = torch_dtype
        self.patch_size = tuple(config.patch_size)
        self.gradient_accumulation_steps = 1


class LingbotVAActionModel(WanTransformer3DModel, BasePolicy):
    def __init__(self, cfg: Any, torch_dtype: torch.dtype = torch.bfloat16):
        self.rlinf_config = cfg
        self.torch_dtype = torch_dtype
        self.repo_path = Path(getattr(cfg.lingbotva, "repo_path"))
        self.model_path = Path(cfg.model_path)
        _extend_import_path(self.repo_path)

        self._sft_enabled = bool(getattr(cfg.lingbotva, "enable_sft", False))
        if self._sft_enabled:
            transformer_path = self.model_path / "transformer"
            WanTransformer3DModel.__init__(
                self,
                **_load_transformer_config(transformer_path),
            )
            self._load_pretrained_transformer(transformer_path)
            self.to(dtype=torch_dtype)
            self.train()
            self.requires_grad_(True)
        else:
            transformer_path = self.model_path / "transformer"
            WanTransformer3DModel.__init__(
                self,
                **_load_transformer_config(transformer_path),
            )
            self._load_pretrained_transformer(transformer_path)
            self.to(dtype=torch_dtype)
            self.eval()
            self.requires_grad_(False)

        self.num_action_chunks = int(getattr(cfg, "num_action_chunks", 32))
        self.action_dim = int(getattr(cfg, "action_dim", 16))
        self.action_per_frame = int(getattr(cfg.lingbotva, "action_per_frame", 16))
        self._episode_states: dict[int, LingbotVAEpisodeState] = {}
        self._runtime_initialized = False
        self._ac_applied = False

        if self._sft_enabled:
            self._init_sft_helpers()
        else:
            atexit.register(self.close)

    def _load_pretrained_transformer(self, transformer_path: Path) -> None:
        if not transformer_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA transformer path not found at {transformer_path}."
            )
        pretrained = WanTransformer3DModel.from_pretrained(
            str(transformer_path),
            torch_dtype=self.torch_dtype,
        )
        missing_keys, unexpected_keys = self.load_state_dict(
            pretrained.state_dict(),
            strict=False,
        )
        del pretrained
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "LingBot-VA transformer weights do not match official model "
                f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
            )

    def _init_sft_helpers(self) -> None:
        from wan_va.configs import VA_CONFIGS
        from wan_va.train import Trainer as WanVATrainer
        from wan_va.utils import FlowMatchScheduler

        train_config_name = getattr(
            self.rlinf_config.lingbotva, "train_config_name", "robotwin_train"
        )
        self.train_cfg = copy.deepcopy(VA_CONFIGS[train_config_name])
        self.train_cfg.wan22_pretrained_model_name_or_path = str(self.model_path)
        self.train_cfg.param_dtype = self.torch_dtype
        self._sft_context = _WanVATrainingContext(
            config=self.train_cfg,
            torch_dtype=self.torch_dtype,
        )
        self._sft_context._add_noise = WanVATrainer._add_noise.__get__(
            self._sft_context,
            type(self._sft_context),
        )
        self._sft_context._prepare_input_dict = (
            WanVATrainer._prepare_input_dict.__get__(
                self._sft_context,
                type(self._sft_context),
            )
        )
        self._sft_context.compute_loss = WanVATrainer.compute_loss.__get__(
            self._sft_context,
            type(self._sft_context),
        )
        self._apply_official_activation_checkpointing()
        self._sft_context.train_scheduler_latent = FlowMatchScheduler(
            shift=self.train_cfg.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self._sft_context.train_scheduler_latent.set_timesteps(1000, training=True)
        self._sft_context.train_scheduler_action = FlowMatchScheduler(
            shift=self.train_cfg.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self._sft_context.train_scheduler_action.set_timesteps(1000, training=True)

    def _apply_official_activation_checkpointing(self) -> None:
        if self._ac_applied:
            return
        from wan_va.distributed.fsdp import apply_ac

        apply_ac(self)
        self._ac_applied = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        del kwargs
        if self._sft_enabled:
            self._apply_official_activation_checkpointing()

    def gradient_checkpointing_disable(self) -> None:
        return None

    def forward(self, forward_type=ForwardType.DEFAULT, *args, **kwargs):
        if not isinstance(forward_type, ForwardType):
            return WanTransformer3DModel.forward(self, forward_type, *args, **kwargs)
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        del kwargs
        if not self._sft_enabled:
            raise NotImplementedError(
                "LingBot-VA SFT support is disabled for the current config."
            )
        first_tensor = next(value for value in data.values() if torch.is_tensor(value))
        device = first_tensor.device
        self._sft_context.device = torch.device(device)
        batch = {
            "latents": data["latents"].to(device=device, dtype=self.torch_dtype),
            "text_emb": data["text_emb"].to(device=device, dtype=self.torch_dtype),
            "actions": data["actions"].to(device=device, dtype=self.torch_dtype),
            "actions_mask": data["actions_mask"].to(device=device),
        }
        input_dict = self._sft_context._prepare_input_dict(batch)
        pred = WanTransformer3DModel.forward(self, input_dict, train_mode=True)
        latent_loss, action_loss = self._sft_context.compute_loss(input_dict, pred)
        total_loss = latent_loss + action_loss
        return {
            "loss": total_loss,
            "latent_loss": latent_loss,
            "action_loss": action_loss,
        }

    def post_sft_checkpoint_save(self, *, save_path: str, rank: int = 0) -> None:
        if rank != 0:
            return

        state_dict_path = os.path.join(save_path, "model_state_dict", "full_weights.pt")
        export_official_transformer_checkpoint(
            model_path=self.model_path,
            state_dict_path=state_dict_path,
            output_dir=os.path.join(save_path, "transformer"),
        )

    def default_forward(self, **kwargs):
        del kwargs
        raise NotImplementedError(
            "LingBot-VA default_forward is not supported in the current eval/SFT integration. "
            "Use predict_action_batch for evaluation and sft_forward for supervised fine-tuning."
        )

    @staticmethod
    def _get_extra_obs(env_obs: dict[str, Any]) -> dict[str, Any]:
        extra_obs = env_obs.get("extra_obs")
        if isinstance(extra_obs, dict):
            return extra_obs
        if isinstance(extra_obs, (list, tuple)):
            merged: dict[str, Any] = {}
            for item in extra_obs:
                if item is None:
                    continue
                if not isinstance(item, dict):
                    raise TypeError(
                        "LingBot-VA expects extra_obs shards to be dict or None, "
                        f"got {type(item)!r}."
                    )
                for key, value in item.items():
                    if key not in merged:
                        merged[key] = value
                        continue
                    existing = merged[key]
                    if isinstance(existing, list):
                        if isinstance(value, list):
                            existing.extend(value)
                        else:
                            existing.append(value)
                    elif isinstance(existing, tuple):
                        if isinstance(value, tuple):
                            merged[key] = existing + value
                        else:
                            merged[key] = existing + (value,)
                    elif torch.is_tensor(existing) and torch.is_tensor(value):
                        try:
                            merged[key] = torch.cat([existing, value], dim=0)
                        except RuntimeError as exc:
                            raise ValueError(
                                "LingBot-VA extra_obs tensor shards must be concatenable "
                                f"for key {key!r}: existing shape {tuple(existing.shape)} "
                                f"vs new shape {tuple(value.shape)}."
                            ) from exc
                    else:
                        merged[key] = value
            return merged
        if extra_obs is not None:
            raise TypeError(
                "LingBot-VA expects extra_obs to be a dict, list/tuple of dict shards, "
                f"or None, got {type(extra_obs)!r}."
            )
        return {}

    def _get_env_meta(self, env_obs: dict[str, Any], key: str) -> Any:
        extra_obs = self._get_extra_obs(env_obs)
        if key in extra_obs:
            return extra_obs[key]
        return env_obs.get(key)

    def _validate_runtime_paths(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA repo path does not exist: {self.repo_path}"
            )
        if not self.repo_path.is_dir():
            raise NotADirectoryError(
                f"LingBot-VA repo path is not a directory: {self.repo_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA model path does not exist: {self.model_path}"
            )
        if not self.model_path.is_dir():
            raise NotADirectoryError(
                f"LingBot-VA model path is not a directory: {self.model_path}"
            )

    @staticmethod
    def _validate_attn_mode(model_path: Path) -> None:
        config_data = _load_transformer_config(model_path / "transformer")
        attn_mode = config_data.get("attn_mode")
        if attn_mode not in {"torch", "flashattn"}:
            raise ValueError(
                "LingBot-VA inference requires transformer/config.json attn_mode "
                f'to be "torch" or "flashattn", but got {attn_mode!r}.'
            )

    def _ensure_runtime(self):
        if self._runtime_initialized:
            return self

        from wan_va.configs import VA_CONFIGS
        from wan_va.modules.utils import (
            WanVAEStreamingWrapper,
            load_text_encoder,
            load_tokenizer,
            load_vae,
        )
        from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id

        self._validate_runtime_paths()
        self._validate_attn_mode(self.model_path)

        config_name = getattr(self.rlinf_config.lingbotva, "config_name", "robotwin")
        self.job_config = copy.deepcopy(VA_CONFIGS[config_name])
        self.job_config.wan22_pretrained_model_name_or_path = str(self.model_path)
        self.job_config.param_dtype = self.torch_dtype
        self.job_config.enable_offload = bool(
            getattr(self.rlinf_config.lingbotva, "enable_offload", False)
        )
        self.job_config.save_root = str(
            _resolve_runtime_path(
                getattr(
                    self.rlinf_config.lingbotva,
                    "save_root",
                    "./runtime/lingbotva",
                )
            )
        )

        current_device = _resolve_cuda_local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(current_device)
        self.job_config.rank = 0
        self.job_config.local_rank = current_device
        self.job_config.world_size = 1

        self.cache_name = "rlinf_batch"
        self.save_root = self.job_config.save_root
        runtime_dtype = self.job_config.param_dtype
        runtime_device = torch.device(f"cuda:{current_device}")
        self.to(dtype=runtime_dtype, device=runtime_device)
        self.eval()
        self.requires_grad_(False)
        self._load_eval_transformer_state_dict(self)

        self.enable_offload = bool(getattr(self.job_config, "enable_offload", True))
        self.scheduler = FlowMatchScheduler(
            shift=self.job_config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        model_root = Path(self.job_config.wan22_pretrained_model_name_or_path)
        aux_device = "cpu" if self.enable_offload else runtime_device
        self.vae = load_vae(
            model_root / "vae",
            torch_dtype=runtime_dtype,
            torch_device=aux_device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        self.tokenizer = load_tokenizer(model_root / "tokenizer")
        self.text_encoder = load_text_encoder(
            model_root / "text_encoder",
            torch_dtype=runtime_dtype,
            torch_device=aux_device,
        )
        self.env_type = self.job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == "robotwin_tshape":
            vae_half = load_vae(
                model_root / "vae",
                torch_dtype=runtime_dtype,
                torch_device=aux_device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        self.eval()
        self.requires_grad_(False)
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id
        self._runtime_initialized = True
        logger.info(
            "Initialized in-process LingBot-VA runtime on device %s "
            "(CUDA_VISIBLE_DEVICES=%s, LOCAL_ACCELERATOR_RANK=%s).",
            current_device,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ.get("LOCAL_ACCELERATOR_RANK"),
        )
        return self

    def _load_eval_transformer_state_dict(self, transformer: torch.nn.Module) -> None:
        transformer_state_dict_path = getattr(
            self.rlinf_config.lingbotva, "transformer_state_dict_path", None
        )
        if transformer_state_dict_path is None:
            return

        checkpoint_path = Path(transformer_state_dict_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "LingBot-VA transformer state dict path does not exist: "
                f"{checkpoint_path}"
            )
        if checkpoint_path.is_dir():
            transformer_dir = (
                checkpoint_path / "transformer"
                if (checkpoint_path / "transformer" / "config.json").exists()
                else checkpoint_path
            )
            state_path = transformer_dir / "diffusion_pytorch_model.safetensors"
            if not state_path.exists():
                raise FileNotFoundError(
                    "LingBot-VA transformer checkpoint directory must contain "
                    f"diffusion_pytorch_model.safetensors: {transformer_dir}"
                )
            transformer_state = load_file(str(state_path), device="cpu")
        else:
            raw_state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(raw_state, dict):
                raise TypeError(
                    "LingBot-VA transformer state dict must deserialize to a dict, got "
                    f"{type(raw_state)!r}."
                )
            transformer_state = _extract_transformer_state_dict(raw_state)

        missing_keys, unexpected_keys = transformer.load_state_dict(
            transformer_state,
            strict=False,
        )
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "LingBot-VA transformer checkpoint does not match the runtime "
                f"model (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})."
            )
        logger.info(
            "Loaded LingBot-VA transformer checkpoint from %s (missing=%d, unexpected=%d).",
            checkpoint_path,
            len(missing_keys),
            len(unexpected_keys),
        )

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        device = device or self.device
        dtype = dtype or self.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(item) for item in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        mask = text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(
            text_input_ids.to(text_encoder_device),
            mask.to(text_encoder_device),
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [
            item[:seq_len] for item, seq_len in zip(prompt_embeds, seq_lens)
        ]
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [
                        item,
                        item.new_zeros(
                            max_sequence_length - item.size(0), item.size(1)
                        ),
                    ]
                )
                for item in prompt_embeds
            ],
            dim=0,
        )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt,
            seq_len,
            -1,
        )
        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length: int = 226,
        device=None,
        dtype=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = device or self.device
        dtype = dtype or self.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    "`negative_prompt` should be the same type as `prompt`, "
                    f"got {type(negative_prompt)} vs {type(prompt)}."
                )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, "
                    f"but `prompt` has batch size {batch_size}."
                )
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def normalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        return ((latents.float() - latents_mean) * latents_std).to(latents)

    def preprocess_action(self, action: np.ndarray) -> torch.Tensor:
        action_model_input = torch.from_numpy(action)
        action_model_input = F.pad(
            action_model_input,
            [0, 0, 0, 0, 0, 1],
            mode="constant",
            value=0,
        )
        action_model_input = action_model_input[
            self.job_config.inverse_used_action_channel_ids
        ]
        if self.action_norm_method == "quantiles":
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6
            ) * 2.0 - 1.0
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)

    def _prepare_latent_input(
        self,
        latent_model_input: torch.Tensor | None,
        action_model_input: torch.Tensor | None,
        latent_t: float = 0,
        action_t: float = 0,
        latent_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        frame_st_id: int = 0,
        patch_size=(1, 2, 2),
    ) -> dict[str, dict[str, torch.Tensor]]:
        input_dict: dict[str, dict[str, torch.Tensor]] = {}
        if latent_model_input is not None:
            input_dict["latent_res_lst"] = {
                "noisy_latents": latent_model_input,
                "timesteps": torch.ones(
                    [latent_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * latent_t,
                "grid_id": self._get_mesh_id(
                    latent_model_input.shape[-3] // patch_size[0],
                    latent_model_input.shape[-2] // patch_size[1],
                    latent_model_input.shape[-1] // patch_size[2],
                    0,
                    1,
                    frame_st_id,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict["latent_res_lst"]["noisy_latents"][:, :, 0:1] = latent_cond[
                    :, :, 0:1
                ]
                input_dict["latent_res_lst"]["timesteps"][0:1] *= 0

        if action_model_input is not None:
            input_dict["action_res_lst"] = {
                "noisy_latents": action_model_input,
                "timesteps": torch.ones(
                    [action_model_input.shape[2]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * action_t,
                "grid_id": self._get_mesh_id(
                    action_model_input.shape[-3],
                    action_model_input.shape[-2],
                    action_model_input.shape[-1],
                    1,
                    1,
                    frame_st_id,
                    action=True,
                ).to(self.device),
                "text_emb": self.prompt_embeds.to(self.dtype).clone(),
            }
            if action_cond is not None:
                input_dict["action_res_lst"]["noisy_latents"][:, :, 0:1] = action_cond[
                    :, :, 0:1
                ]
                input_dict["action_res_lst"]["timesteps"][0:1] *= 0
            input_dict["action_res_lst"]["noisy_latents"][:, ~self.action_mask] *= 0
        return input_dict

    def _cfg_batch_size(self, batch_size: int) -> int:
        runtime = self._ensure_runtime()
        return batch_size * (2 if runtime.use_cfg else 1)

    def _ensure_observation_runtime_shape(self) -> None:
        runtime = self._ensure_runtime()
        if all(
            hasattr(runtime, attr)
            for attr in ("height", "width", "latent_height", "latent_width")
        ):
            return

        runtime.action_per_frame = runtime.job_config.action_per_frame
        runtime.height, runtime.width = (
            runtime.job_config.height,
            runtime.job_config.width,
        )
        if runtime.env_type == "robotwin_tshape":
            runtime.latent_height, runtime.latent_width = (
                ((runtime.height // 16) * 3) // 2,
                runtime.width // 16,
            )
        else:
            runtime.latent_height, runtime.latent_width = (
                runtime.height // 16,
                runtime.width // 16 * len(runtime.job_config.obs_cam_keys),
            )

    def _reset_batch_runtime(self, prompts: list[str]) -> None:
        if not prompts:
            raise ValueError("LingBot-VA batch runtime requires at least one prompt.")

        runtime = self._ensure_runtime()
        batch_size = len(prompts)
        runtime.cache_name = "rlinf_batch"
        runtime.use_cfg = (runtime.job_config.guidance_scale > 1) or (
            runtime.job_config.action_guidance_scale > 1
        )
        runtime.frame_st_id = 0
        runtime.init_latent = None
        runtime.clear_cache(runtime.cache_name)
        runtime.streaming_vae.clear_cache()
        self._ensure_observation_runtime_shape()

        if runtime.env_type == "robotwin_tshape":
            runtime.streaming_vae_half.clear_cache()

        patch_size = runtime.job_config.patch_size
        latent_token_per_chunk = (
            runtime.job_config.frame_chunk_size
            * runtime.latent_height
            * runtime.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = (
            runtime.job_config.frame_chunk_size * runtime.action_per_frame
        )
        runtime.create_empty_cache(
            runtime.cache_name,
            runtime.job_config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=runtime.dtype,
            device=runtime.device,
            batch_size=self._cfg_batch_size(batch_size),
        )

        runtime.action_mask = torch.zeros([runtime.job_config.action_dim]).bool()
        runtime.action_mask[runtime.job_config.used_action_channel_ids] = True
        runtime.actions_q01 = torch.tensor(
            runtime.job_config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        runtime.actions_q99 = torch.tensor(
            runtime.job_config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        runtime.action_norm_method = runtime.job_config.action_norm_method
        runtime.prompt_embeds, runtime.negative_prompt_embeds = self.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=runtime.job_config.guidance_scale > 1,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        runtime.exp_name = "rlinf_batch"
        runtime.exp_save_root = str(Path(runtime.save_root) / "real")
        os.makedirs(runtime.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    @staticmethod
    def _normalize_obs_sequences(
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        if not obs_batch:
            raise ValueError("LingBot-VA batch infer requires non-empty observations.")
        if isinstance(obs_batch[0], dict):
            return [[obs] for obs in obs_batch]
        return obs_batch  # type: ignore[return-value]

    @staticmethod
    def _validate_uniform_sequence_lengths(
        obs_sequences: list[list[dict[str, Any]]],
        *,
        context: str,
    ) -> None:
        lengths = [len(sequence) for sequence in obs_sequences]
        if not lengths:
            raise ValueError(f"{context} requires non-empty observation sequences.")
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{context} requires uniform sequence lengths across the batch, got {lengths}."
            )

    def _encode_obs_batch(
        self,
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> torch.Tensor | None:
        runtime = self._ensure_runtime()
        self._ensure_observation_runtime_shape()
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        if not obs_sequences:
            return None
        self._validate_uniform_sequence_lengths(
            obs_sequences,
            context="LingBot-VA batched observation encoding",
        )

        videos = []
        for camera_index, camera_key in enumerate(runtime.job_config.obs_cam_keys):
            if runtime.env_type == "robotwin_tshape":
                if camera_index == 0:
                    height_i, width_i = runtime.height, runtime.width
                else:
                    height_i, width_i = runtime.height // 2, runtime.width // 2
            else:
                height_i, width_i = runtime.height, runtime.width

            history_video = torch.stack(
                [
                    torch.from_numpy(np.stack([frame[camera_key] for frame in seq]))
                    .float()
                    .permute(3, 0, 1, 2)
                    for seq in obs_sequences
                ],
                dim=0,
            )
            history_video = F.interpolate(
                history_video.flatten(0, 1),
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (history_video.shape[0], history_video.shape[1]))
            videos.append(history_video)

        if runtime.env_type == "robotwin_tshape":
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(runtime.streaming_vae.vae.parameters()).device
            enc_out_high = runtime.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(runtime.dtype)
            )
            enc_out_left_and_right = runtime.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(runtime.dtype)
            )
            left_enc, right_enc = enc_out_left_and_right.split(
                videos_high.shape[0], dim=0
            )
            enc_out = torch.cat(
                [
                    torch.cat([left_enc, right_enc], dim=-1),
                    enc_out_high,
                ],
                dim=-2,
            )
        else:
            videos_tensor = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(runtime.streaming_vae.vae.parameters()).device
            enc_out = runtime.streaming_vae.encode_chunk(
                videos_tensor.to(vae_device).to(runtime.dtype)
            )

        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(runtime.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(runtime.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        if runtime.env_type == "robotwin_tshape":
            video_latent = mu_norm.to(runtime.device)
        else:
            video_latent = torch.cat(
                mu_norm.split(len(obs_sequences), dim=0), dim=-1
            ).to(runtime.device)
        return video_latent

    def _preprocess_action_batch(self, state_batch: np.ndarray) -> torch.Tensor:
        self._ensure_runtime()
        tensors = [
            self.preprocess_action(np.asarray(state, dtype=np.float32))
            for state in state_batch
        ]
        return torch.cat(tensors, dim=0)

    def _repeat_official_input_for_batch_cfg(
        self, input_value: dict[str, torch.Tensor], batch_size: int
    ) -> dict[str, torch.Tensor]:
        runtime = self._ensure_runtime()
        if runtime.use_cfg:
            input_value["noisy_latents"] = input_value["noisy_latents"].repeat(
                2, 1, 1, 1, 1
            )
            input_value["text_emb"] = torch.cat(
                [
                    runtime.prompt_embeds.to(runtime.dtype).clone(),
                    runtime.negative_prompt_embeds.to(runtime.dtype).clone(),
                ],
                dim=0,
            )
        input_value["grid_id"] = input_value["grid_id"][None].repeat(
            self._cfg_batch_size(batch_size), 1, 1
        )
        input_value["timesteps"] = input_value["timesteps"][None].repeat(
            self._cfg_batch_size(batch_size), 1
        )
        return input_value

    def _prepare_batch_input(
        self,
        *,
        latent_model_input: torch.Tensor | None,
        action_model_input: torch.Tensor | None,
        latent_t: float = 0,
        action_t: float = 0,
        latent_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        frame_st_id: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        runtime = self._ensure_runtime()
        batch_size = (
            latent_model_input.shape[0]
            if latent_model_input is not None
            else action_model_input.shape[0]
        )
        input_dict = self._prepare_latent_input(
            latent_model_input,
            action_model_input,
            latent_t=latent_t,
            action_t=action_t,
            latent_cond=latent_cond,
            action_cond=action_cond,
            frame_st_id=frame_st_id,
            patch_size=runtime.job_config.patch_size,
        )
        return {
            key: self._repeat_official_input_for_batch_cfg(value, batch_size)
            for key, value in input_dict.items()
        }

    def _postprocess_action_batch(self, action: torch.Tensor) -> list[np.ndarray]:
        runtime = self._ensure_runtime()
        action = action.detach().cpu()[..., 0]
        if runtime.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (
                runtime.actions_q99 - runtime.actions_q01 + 1e-6
            ) + runtime.actions_q01
        else:
            raise NotImplementedError
        action_np = action.numpy()
        used = action_np[:, runtime.job_config.used_action_channel_ids]
        return [used[idx].astype(np.float32) for idx in range(used.shape[0])]

    def _infer_batch_impl(
        self,
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        frame_st_id: int = 0,
    ) -> list[np.ndarray]:
        runtime = self._ensure_runtime()
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        batch_size = len(obs_sequences)

        if frame_st_id == 0 and runtime.init_latent is None:
            runtime.init_latent = self._encode_obs_batch(obs_sequences)

        latents = torch.randn(
            batch_size,
            48,
            runtime.job_config.frame_chunk_size,
            runtime.latent_height,
            runtime.latent_width,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        actions = torch.randn(
            batch_size,
            runtime.job_config.action_dim,
            runtime.job_config.frame_chunk_size,
            runtime.action_per_frame,
            1,
            device=runtime.device,
            dtype=runtime.dtype,
        )

        runtime.scheduler.set_timesteps(runtime.job_config.num_inference_steps)
        runtime.action_scheduler.set_timesteps(
            runtime.job_config.action_num_inference_steps
        )
        timesteps = F.pad(runtime.scheduler.timesteps, (0, 1), mode="constant", value=0)
        if runtime.job_config.video_exec_step != -1:
            timesteps = timesteps[: runtime.job_config.video_exec_step]
        action_timesteps = F.pad(
            runtime.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )

        with torch.no_grad():
            for step_idx, timestep in enumerate(timesteps):
                last_step = step_idx == len(timesteps) - 1
                latent_cond = (
                    runtime.init_latent[:, :, 0:1] if frame_st_id == 0 else None
                )
                input_dict = self._prepare_batch_input(
                    latent_model_input=latents,
                    action_model_input=None,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=latent_cond,
                    action_cond=None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = runtime(
                    input_dict["latent_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=runtime.cache_name,
                    action_mode=False,
                )
                if not last_step or runtime.job_config.video_exec_step != -1:
                    if self._data_seq_to_patch is None:
                        raise RuntimeError(
                            "LingBot-VA batch runtime missing data_seq_to_patch helper."
                        )
                    video_noise_pred = self._data_seq_to_patch(
                        runtime.job_config.patch_size,
                        video_noise_pred,
                        runtime.job_config.frame_chunk_size,
                        runtime.latent_height,
                        runtime.latent_width,
                        batch_size=self._cfg_batch_size(batch_size),
                    )
                    if runtime.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[
                            batch_size:
                        ] + runtime.job_config.guidance_scale * (
                            video_noise_pred[:batch_size]
                            - video_noise_pred[batch_size:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:batch_size]
                    latents = runtime.scheduler.step(
                        video_noise_pred, timestep, latents, return_dict=False
                    )
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond

            for step_idx, timestep in enumerate(action_timesteps):
                last_step = step_idx == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [
                            batch_size,
                            runtime.job_config.action_dim,
                            1,
                            runtime.action_per_frame,
                            1,
                        ],
                        device=runtime.device,
                        dtype=runtime.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_batch_input(
                    latent_model_input=None,
                    action_model_input=actions,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=None,
                    action_cond=action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = runtime(
                    input_dict["action_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=runtime.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = (
                        action_noise_pred.unflatten(
                            1,
                            (
                                runtime.job_config.frame_chunk_size,
                                runtime.action_per_frame,
                            ),
                        )
                        .permute(0, 3, 1, 2)
                        .unsqueeze(-1)
                    )
                    if runtime.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[
                            batch_size:
                        ] + runtime.job_config.action_guidance_scale * (
                            action_noise_pred[:batch_size]
                            - action_noise_pred[batch_size:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:batch_size]
                    actions = runtime.action_scheduler.step(
                        action_noise_pred, timestep, actions, return_dict=False
                    )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~runtime.action_mask] *= 0
        torch.cuda.empty_cache()
        return self._postprocess_action_batch(actions)

    def _compute_kv_cache_batch_impl(
        self,
        *,
        obs_batch: list[list[dict[str, Any]]],
        state_batch: np.ndarray,
        frame_st_id: int,
    ) -> int:
        runtime = self._ensure_runtime()
        runtime.clear_pred_cache(runtime.cache_name)
        latent_model_input = self._encode_obs_batch(obs_batch)
        if frame_st_id == 0:
            latent_model_input = (
                torch.cat([runtime.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else runtime.init_latent
            )

        action_model_input = self._preprocess_action_batch(state_batch).to(
            latent_model_input
        )
        input_dict = self._prepare_batch_input(
            latent_model_input=latent_model_input,
            action_model_input=action_model_input,
            frame_st_id=frame_st_id,
        )

        with torch.no_grad():
            runtime(
                input_dict["latent_res_lst"],
                update_cache=2,
                cache_name=runtime.cache_name,
                action_mode=False,
            )
            runtime(
                input_dict["action_res_lst"],
                update_cache=2,
                cache_name=runtime.cache_name,
                action_mode=True,
            )
        torch.cuda.empty_cache()
        return frame_st_id + int(latent_model_input.shape[2])

    def infer_batch(
        self,
        obs_batch: list[dict[str, Any]],
        prompts: list[str],
        *,
        kv_cache_histories: list[list[tuple[list[dict[str, Any]], np.ndarray]]]
        | None = None,
    ) -> list[np.ndarray]:
        if len(obs_batch) != len(prompts):
            raise ValueError(
                f"LingBot-VA batch infer expects equal numbers of obs and prompts, got {len(obs_batch)} and {len(prompts)}."
            )
        self._reset_batch_runtime(prompts)
        frame_st_id = 0
        if kv_cache_histories:
            self.init_latent = self._encode_obs_batch(obs_batch)
            history_len = len(kv_cache_histories[0])
            if not all(len(history) == history_len for history in kv_cache_histories):
                raise ValueError(
                    "LingBot-VA batched follow-up infer requires groups with equal kv history lengths."
                )
            for history_idx in range(history_len):
                obs_sequences = [
                    history[history_idx][0] for history in kv_cache_histories
                ]
                self._validate_uniform_sequence_lengths(
                    obs_sequences,
                    context=(
                        f"LingBot-VA batched follow-up infer replay step {history_idx}"
                    ),
                )
                state_batch = np.stack(
                    [history[history_idx][1] for history in kv_cache_histories], axis=0
                )
                try:
                    frame_st_id = self._compute_kv_cache_batch_impl(
                        obs_batch=obs_sequences,
                        state_batch=state_batch,
                        frame_st_id=frame_st_id,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "LingBot-VA batched follow-up kv-cache replay failed at "
                        f"history_idx={history_idx}, "
                        f"obs_lengths={[len(seq) for seq in obs_sequences]}, "
                        f"state_batch_shape={tuple(state_batch.shape)}, "
                        f"frame_st_id={frame_st_id}."
                    ) from exc
        else:
            return self._infer_batch_impl(obs_batch, frame_st_id=frame_st_id)
        return self._infer_batch_impl(obs_batch, frame_st_id=frame_st_id)

    def _get_prompt(self, env_obs: dict[str, Any], env_idx: int) -> str:
        prompts = env_obs.get("task_descriptions")
        if prompts is None:
            raise ValueError(
                "LingBot-VA requires task_descriptions in env observations."
            )
        return str(prompts[env_idx])

    def _get_state(self, env_idx: int) -> LingbotVAEpisodeState:
        if env_idx not in self._episode_states:
            self._episode_states[env_idx] = LingbotVAEpisodeState()
        return self._episode_states[env_idx]

    def _format_initial_eef_pose(
        self, env_obs: dict[str, Any], env_idx: int
    ) -> np.ndarray | None:
        eef_poses = self._get_env_meta(env_obs, "eef_poses")
        if eef_poses is None:
            return None
        if isinstance(eef_poses, torch.Tensor):
            pose = eef_poses[env_idx].detach().cpu().numpy().astype(np.float32)
        else:
            pose = np.asarray(eef_poses[env_idx], dtype=np.float32)
        if pose.shape[-1] != 16:
            raise ValueError(f"Expected 16D reset ee pose, got shape {pose.shape}.")
        return pose.copy()

    @staticmethod
    def _add_eef_pose(delta_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
        delta_rot = R.from_quat(delta_pose[3:7][None])
        init_rot = R.from_quat(init_pose[3:7][None])
        out_rot = (init_rot * delta_rot).as_quat().reshape(-1)
        out_trans = delta_pose[:3] + init_pose[:3]
        return np.concatenate([out_trans, out_rot, delta_pose[7:8]], axis=0)

    def _add_init_pose(
        self, delta_action: np.ndarray, initial_pose: np.ndarray
    ) -> np.ndarray:
        left = self._add_eef_pose(delta_action[:8], initial_pose[:8])
        right = self._add_eef_pose(delta_action[8:], initial_pose[8:])
        out = np.concatenate([left, right], axis=0).astype(np.float32)
        out[3:7] = out[3:7] / np.linalg.norm(out[3:7])
        out[11:15] = out[11:15] / np.linalg.norm(out[11:15])
        return out

    def _select_executable_actions(
        self, raw_action: np.ndarray, first_chunk: bool
    ) -> np.ndarray:
        start_idx = 1 if first_chunk else 0
        selected = raw_action[:, start_idx:, :]
        return np.transpose(selected, (1, 2, 0)).reshape(-1, raw_action.shape[0])

    def _convert_raw_to_env_actions(
        self,
        raw_action: np.ndarray,
        initial_eef_pose: np.ndarray,
        first_chunk: bool,
    ) -> np.ndarray:
        executable_model = self._select_executable_actions(raw_action, first_chunk)
        env_actions = np.stack(
            [self._add_init_pose(step, initial_eef_pose) for step in executable_model],
            axis=0,
        )
        return env_actions

    def _reset_episode(
        self, env_idx: int, prompt: str, env_obs: dict[str, Any]
    ) -> LingbotVAEpisodeState:
        state = self._get_state(env_idx)
        initial_eef_pose = self._format_initial_eef_pose(env_obs, env_idx)
        if initial_eef_pose is None:
            raise ValueError(
                "LingBot-VA requires reset-time eef_poses from RoboTwinEnv."
            )
        state.reset(prompt=prompt, initial_eef_pose=initial_eef_pose)
        return state

    @staticmethod
    def _group_env_indices_for_batch_refill(
        env_indices: list[int],
        episode_states: dict[int, LingbotVAEpisodeState],
    ) -> list[list[int]]:
        grouped: dict[tuple[bool, int], list[int]] = {}
        for env_idx in env_indices:
            state = episode_states[env_idx]
            group_key = (
                state.first_chunk,
                len(state.kv_cache_history),
            )
            grouped.setdefault(group_key, []).append(env_idx)
        return list(grouped.values())

    def _fill_action_queue_from_raw_action(
        self,
        state: LingbotVAEpisodeState,
        raw_action: np.ndarray,
    ) -> None:
        if raw_action.ndim != 3:
            raise ValueError(
                f"LingBot-VA raw action must be 3D, got shape {tuple(raw_action.shape)}."
            )
        if raw_action.shape[2] % 4 != 0:
            raise ValueError(
                "LingBot-VA follow-up key-frame cadence requires raw_action.shape[2] "
                f"to be divisible by 4, got {raw_action.shape[2]}."
            )
        state.prev_model_action = raw_action.astype(np.float32)
        state.last_action_per_frame = raw_action.shape[2] // 4
        env_actions = self._convert_raw_to_env_actions(
            raw_action=raw_action,
            initial_eef_pose=state.initial_eef_pose,
            first_chunk=state.first_chunk,
        )
        state.action_queue.clear()
        for action in env_actions:
            state.action_queue.append(action.astype(np.float32))
        state.first_chunk = False

    def _pop_action_chunk(
        self,
        state: LingbotVAEpisodeState,
        chunk_len: int,
    ) -> np.ndarray:
        if chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {chunk_len}.")
        if len(state.action_queue) < chunk_len:
            raise RuntimeError(
                "LingBot-VA action queue does not have enough buffered steps: "
                f"env_queue={len(state.action_queue)}, requested={chunk_len}."
            )
        env_actions = np.stack(
            [state.action_queue.popleft() for _ in range(chunk_len)], axis=0
        ).astype(np.float32)
        return env_actions

    @staticmethod
    def _history_signature(
        history: list[tuple[list[dict[str, Any]], np.ndarray]],
    ) -> tuple[int, ...]:
        return tuple(len(obs_seq) for obs_seq, _state in history)

    def _refill_action_queue_batch(
        self,
        env_indices: list[int],
        env_obs: dict[str, Any],
    ) -> None:
        prompts = [self._get_prompt(env_obs, env_idx) for env_idx in env_indices]
        states = [self._get_state(env_idx) for env_idx in env_indices]

        if all(state.first_chunk for state in states):
            obs_batch = []
            for env_idx, prompt, state in zip(env_indices, prompts, states):
                first_obs = LingbotVAObservationAdapter.format_observation(
                    env_obs, env_idx, prompt
                )
                state.first_obs = first_obs
                obs_batch.append(first_obs)
            raw_actions = self.infer_batch(obs_batch, prompts)
        else:
            followup_groups: dict[
                tuple[int, ...],
                dict[str, list[Any]],
            ] = {}
            for env_idx, prompt, state in zip(env_indices, prompts, states):
                raw_chunk_observations = self._get_env_meta(
                    env_obs, "chunk_observations"
                )
                if (
                    raw_chunk_observations is None
                    or env_idx >= len(raw_chunk_observations)
                    or not raw_chunk_observations[env_idx]
                ):
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires non-empty raw chunk observations."
                    )
                if state.first_obs is None:
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires cached first_obs from the first infer."
                    )
                if state.prev_model_action is None:
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires cached previous model action."
                    )
                action_per_frame = state.last_action_per_frame or self.action_per_frame
                key_frames = select_key_frames(
                    chunk_observations=raw_chunk_observations,
                    env_idx=env_idx,
                    prompt=prompt,
                    action_per_frame=action_per_frame,
                )
                if not key_frames:
                    raise ValueError(
                        "LingBot-VA follow-up chunk requires non-empty key_frame_list."
                    )
                history_entry = (key_frames, state.prev_model_action.copy())
                full_history = state.kv_cache_history + [history_entry]
                signature = self._history_signature(full_history)
                group = followup_groups.setdefault(
                    signature,
                    {
                        "obs_batch": [],
                        "prompts": [],
                        "history_batch": [],
                        "states": [],
                        "pending_entries": [],
                    },
                )
                group["obs_batch"].append(state.first_obs)
                group["prompts"].append(prompt)
                group["history_batch"].append(full_history)
                group["states"].append(state)
                group["pending_entries"].append(history_entry)

            for group in followup_groups.values():
                raw_actions = self.infer_batch(
                    group["obs_batch"],
                    group["prompts"],
                    kv_cache_histories=group["history_batch"],
                )
                for state, history_entry, raw_action in zip(
                    group["states"],
                    group["pending_entries"],
                    raw_actions,
                ):
                    state.kv_cache_history.append(history_entry)
                    self._fill_action_queue_from_raw_action(state, raw_action)
            return

        for state, raw_action in zip(states, raw_actions):
            self._fill_action_queue_from_raw_action(state, raw_action)

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **_: Any
    ):
        if mode != "eval":
            raise NotImplementedError(
                "LingBot-VA predict_action_batch only supports eval mode in the current integration."
            )
        states_tensor = env_obs.get("states")
        if states_tensor is None:
            raise ValueError("LingBot-VA requires batched states in env observations.")
        batch_size = states_tensor.shape[0]

        actions = []
        episode_dones = self._get_env_meta(env_obs, "episode_dones")
        pending_refill_envs: list[int] = []
        for env_idx in range(batch_size):
            prompt = self._get_prompt(env_obs, env_idx)
            episode_done = False
            if isinstance(episode_dones, torch.Tensor):
                if episode_dones.dim() == 1:
                    episode_done = bool(episode_dones[env_idx].item())
                else:
                    episode_done = bool(episode_dones[env_idx, -1].item())

            state = self._get_state(env_idx)
            if state.prompt != prompt or state.prompt is None or episode_done:
                state = self._reset_episode(env_idx, prompt, env_obs)
            if not state.action_queue:
                pending_refill_envs.append(env_idx)

        for refill_group in self._group_env_indices_for_batch_refill(
            pending_refill_envs, self._episode_states
        ):
            self._refill_action_queue_batch(refill_group, env_obs)

        available_chunk_lengths = []
        for env_idx in range(batch_size):
            state = self._get_state(env_idx)
            if not state.action_queue:
                raise RuntimeError("LingBot-VA refill produced an empty action queue.")
            available_chunk_lengths.append(len(state.action_queue))

        common_chunk_len = min(available_chunk_lengths)
        if common_chunk_len <= 0:
            raise RuntimeError(
                "LingBot-VA could not determine a positive shared chunk length: "
                + str(available_chunk_lengths)
            )

        for env_idx in range(batch_size):
            state = self._get_state(env_idx)
            chunk_actions = self._pop_action_chunk(state, common_chunk_len)
            actions.append(torch.from_numpy(chunk_actions))

        action_tensor = torch.stack(actions, dim=0).to(dtype=torch.float32)
        zeros = torch.zeros(action_tensor.shape[:2], dtype=torch.float32)
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {"action": action_tensor},
        }
        return action_tensor, result

    def close(self) -> None:
        if not getattr(self, "_runtime_initialized", False):
            return
        try:
            self.clear_cache("rlinf_batch")
        except Exception:
            pass
        streaming_vae = getattr(self, "streaming_vae", None)
        if streaming_vae is not None:
            streaming_vae.clear_cache()
        streaming_vae_half = getattr(self, "streaming_vae_half", None)
        if streaming_vae_half is not None:
            streaming_vae_half.clear_cache()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
