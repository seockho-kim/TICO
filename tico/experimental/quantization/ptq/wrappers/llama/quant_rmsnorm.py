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

from typing import Iterable, Optional

import torch

from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import try_register


@try_register("transformers.models.llama.modeling_llama.LlamaRMSNorm")
class QuantLlamaRMSNorm(QuantModuleBase):
    """
    QuantLlamaRMSNorm — drop-in replacement for LlamaRMSNorm
    Observers:
        act_in, weight, act_out
    """

    def __init__(
        self,
        fp: LlamaRMSNorm,
        *,
        qcfg: Optional[QuantConfig] = None,
        fp_name: Optional[str] = None
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp

        self.act_in_obs = self._make_obs("act_in", dtype=DType.int(16))
        self.act_out_obs = self._make_obs("act_out", dtype=DType.int(16))
        self.weight_obs = self._make_obs("weight", dtype=DType.int(16))

    def enable_calibration(self) -> None:
        """
        Switch to CALIB mode and collect *fixed* ranges for affine params
        immediately, since they do not change across inputs.
        """
        super().enable_calibration()
        self.weight_obs.collect(self.module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._fq(x, self.act_in_obs)

        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.weight_obs.fake_quant(w)

        # Fused RMSNorm operation
        y = torch.ops.circle_custom.rms_norm(x_q, w, self.module.variance_epsilon)

        return self._fq(y, self.act_out_obs)

    def _all_observers(self) -> Iterable:
        return (self.act_in_obs, self.act_out_obs, self.weight_obs)
