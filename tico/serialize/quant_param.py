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

"""
This is a key for torch.fx.Node's meta dict to save QuantParam

QuantParam can be retrieved as node.meta[QPARAM_KEY]
"""
QPARAM_KEY = "_quantization_parameters_"

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class QuantParam:
    scale: Optional[List[float]] = None
    zero_point: Optional[List[int]] = None
    quantized_dimension: Optional[int] = None
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None
    # NOTE We define dtype as a string to easily extend new dtypes (ex: uint4)
    dtype: str = ""


def to_qparam_dtype(dtype: torch.dtype) -> str:
    str_type = str(dtype)
    assert str_type.startswith("torch.")
    return str_type[6:]
