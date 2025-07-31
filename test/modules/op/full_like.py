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


class SimpleFullLike(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.full_like(x, 1.0)
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class SimpleFullLikeWithDtypeIntToBool(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_bool = torch.full_like(x, 0, dtype=torch.bool)
        z = torch.logical_and(x_bool, y)
        return z

    def get_example_inputs(self):
        return (torch.rand(2, 3).to(dtype=torch.int32), torch.rand(2, 3) > 0.5), {}


class SimpleFullLikeWithDtypeBoolToInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_int = torch.full_like(x, 0, dtype=torch.int32)
        z = x_int + y
        return z

    def get_example_inputs(self):
        return (torch.rand(2, 3) > 0.5, torch.rand(2, 3).to(dtype=torch.int32)), {}


class SimpleFullLikeWithDtypeIntToFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_float = torch.full_like(x, 0, dtype=torch.float32)
        z = x_float + y
        return z

    def get_example_inputs(self):
        return (torch.rand(2, 3).to(dtype=torch.int32), torch.rand(2, 3)), {}


class SimpleFullLikeBool(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.full_like(x, True)
        z = torch.logical_and(x, y)
        return z

    def get_example_inputs(self):
        return (torch.rand(2, 3) > 0.5,), {}


class TrivialDtypeKwargsExample(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_bool = torch.full_like(x, 0, dtype=torch.bool)
        z = torch.logical_and(x_bool, y)
        return z

    def get_example_inputs(self):
        return (torch.rand(2, 3) > 0.5, torch.rand(2, 3) > 0.5), {}
