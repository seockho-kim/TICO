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

import torch
from tico.utils.utils import HAS_TORCH_OVER_28_DEV

from test.utils import tag


class InterpolateDouble(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 4),)


class InterpolateThreeTimes(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=3.0, mode="nearest")

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 4),)


@tag.skip_if(
    not HAS_TORCH_OVER_28_DEV,
    reason="The case isn't supported yet. It will be supported from torch 2.8.0.dev",
)
class InterpolateOnePointFive(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=1.5, mode="nearest")

    def get_example_inputs(self):
        return (torch.randn(1, 3, 6, 6),)
