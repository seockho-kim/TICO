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


class SimpleAnyFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any(dim=0)
        return y

    def get_example_inputs(self):
        x = torch.randn([10])
        return (x,), {}


class SimpleAnyFloat2D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any()
        return y

    def get_example_inputs(self):
        x = torch.randn([2, 3])
        return (x,), {}


class SimpleAnyFloat2DDim2(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any(dim=1)
        return y

    def get_example_inputs(self):
        x = torch.randn([2, 3])
        return (x,), {}


class SimpleAnyBool(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any()
        return y

    def get_example_inputs(self):
        x = torch.randn([10]) > 0.9
        return (x,), {}


class SimpleAnyBool2D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any()
        return y

    def get_example_inputs(self):
        x = torch.randn([2, 3]) > 0.9
        return (x,), {}


class SimpleAnyBool2DKeepDimTrue(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any(keepdim=True)
        return y

    def get_example_inputs(self):
        x = torch.randn([2, 3]) > 0.9
        return (x,), {}


class SimpleAnyInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any()
        return y

    def get_example_inputs(self):
        x = torch.Tensor([[0, 0, 0, 0, 0, 0]]).to(torch.int32)
        return (x,), {}


class SimpleAnyIntMinus(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        y = x.any()
        return y

    def get_example_inputs(self):
        x = torch.Tensor([[0, 0, 0], [0, -1, 0]]).to(torch.int32)
        return (x,), {}
