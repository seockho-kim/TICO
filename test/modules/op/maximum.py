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

from test.modules.base import TestModuleBase


class MaximumWithTwoInputs(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.maximum(x, y)

    def get_example_inputs(self):
        return (torch.randn(1, 3), torch.randn(1, 3)), {}


class MaximumWithIntConstRight(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        right = torch.full_like(x, 0)

        return torch.maximum(x, right)

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class MaximumWithFloatConstRight(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        right = torch.full_like(x, 0.0)

        return torch.maximum(x, right)

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class MaximumWithTensorLeftConstRight(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        left = torch.add(x, 1.0)
        right = torch.full_like(x, 1.0)
        return torch.minimum(left, right)

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}
