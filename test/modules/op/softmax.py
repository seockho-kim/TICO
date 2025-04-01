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


class SimpleSoftMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch._softmax(x, dim=2, half_to_float=False)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(4, 4, 3),)


class SimpleSoftMaxDimMinus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch._softmax(x, dim=-1, half_to_float=False)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(4, 4, 3),)


class SimpleSafeSoftMax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.ops.aten._safe_softmax(x, dim=2)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(4, 4, 3),)
