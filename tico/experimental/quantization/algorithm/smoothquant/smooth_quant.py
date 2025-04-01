# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch


@torch.no_grad()
def smooth_weights(
    front_module: torch.nn.Module,
    back_modules: torch.nn.Module | List[torch.nn.Module],
    activation_max: torch.Tensor,
    alpha: float,
):
    """
    Applies SmoothQuant-style smoothing to the weights and biases of two
     connected modules using activation maximum values.

    NOTE All modules **MUST** have `weight` and optionally `bias` attributes.

    Parameters
    -----------
        front_module
            The front module whose weights and biases will be adjusted.
        back_modules
            A list of back modules whose weights and biases will be adjusted.
        activation_max
            A tensor of channel-wise maximum activation values for the front module.
        alpha
            The smoothing factor that determines the scaling for weight adjustments.

    Raises
    -------
    AttributeError
        If `front_module` or any module in `back_modules` does not have `weight` attributes.
    ValueError
        If the shape of tensors in `activation_max` does not match the number of channels
         in `front_module`'s weight.
    NoteImplementedError
        If `front_module` or any module in `back_modules` is of an unsupported type.
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    if not isinstance(back_modules, list):
        back_modules = [back_modules]

    # Check attributes
    if not hasattr(front_module, "weight"):
        raise AttributeError(
            f"The front module '{type(front_module).__name__}' does not have a 'weight' attribute."
        )
    for back_m in back_modules:
        if not hasattr(back_m, "weight"):
            raise AttributeError(
                f"The front module '{type(back_m).__name__}' does not have a 'weight' attribute."
            )
    # Check shapes
    if isinstance(front_module, LlamaRMSNorm):
        front_numel = front_module.weight.numel()
    else:
        raise NotImplementedError(
            f"Unsupported module type: {type(front_module).__name__}"
        )
    for back_m in back_modules:
        if isinstance(back_m, torch.nn.Linear):
            back_numel = back_m.in_features
        else:
            raise NotImplementedError(
                f"Unsupported module type: {type(front_module).__name__}"
            )

        if front_numel != back_numel or back_numel != activation_max.numel():
            raise ValueError(
                f"Shape mismatch: front_numel({front_numel}), back_numel({back_numel}), activation_max_numel({activation_max.numel()})"
            )

    # Compute scales
    device, dtype = back_modules[0].weight.device, back_modules[0].weight.dtype
    activation_max = activation_max.to(device=device, dtype=dtype)  # type: ignore[arg-type]
    weight_scales = torch.cat(
        [back_m.weight.abs().max(dim=0, keepdim=True)[0] for back_m in back_modules],  # type: ignore[operator]
        dim=0,
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (activation_max.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)  # type: ignore[arg-type]
        .to(dtype)  # type: ignore[arg-type]
    )

    # Smooth
    front_module.weight.div_(scales)
    if hasattr(front_module, "bias"):
        front_module.bias.div_(scales)

    for back_m in back_modules:
        back_m.weight.mul_(scales.view(1, -1))  # type: ignore[operator]


@torch.no_grad()
def apply_smoothing(
    model: torch.nn.Module,
    activation_max: Dict[str, torch.Tensor],
    alpha: float = 0.5,
    custom_alpha_map: Optional[Dict[str, float]] = None,
):
    """
    Applies SmoothQuant-style smoothing to the model's weights using activation maximum values.

    Parameters
    -----------
        model
            A torch module whose weights will be smoothed.
        activation_max
            The channel-wise maximum activation values for the model.
        alpha
            The default smoothing factor to apply across all modules.
        custom_alpha_map
            A dictionary mapping layer/module names to custom alpha values.
            Layers specified in this dictionary will use the corresponding alpha
             value instead of the default.
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    for name, module in model.named_modules():
        alpha_to_apply = alpha
        if custom_alpha_map and name in custom_alpha_map:
            alpha_to_apply = custom_alpha_map[name]
        if alpha_to_apply > 1.0:
            raise RuntimeError(
                f"Alpha value cannot exceed 1.0. Given alpha: {alpha_to_apply}"
            )
        # SmoothQuant is applied before capturing the graph. Therefore, it needs to know
        #  specific module information.
        # TODO Suport more modules.
        if isinstance(module, LlamaDecoderLayer):
            attn_ln = module.input_layernorm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = activation_max[name + ".self_attn.q_proj"]
            smooth_weights(attn_ln, qkv, qkv_input_scales, alpha_to_apply)

            ffn_ln = module.post_attention_layernorm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = activation_max[name + ".mlp.gate_proj"]

            smooth_weights(ffn_ln, fcs, fcs_input_scales, alpha_to_apply)
