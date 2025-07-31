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


class SimpleSub(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_, other):
        return torch.sub(input_, other)

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3)), {}


class SubWithOut(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, input_, other):
        out = torch.empty_like(input_)
        torch.sub(input_, other, out=out)
        return out

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3)), {}


class SubWithBuiltinFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x - y
        return z

    def get_example_inputs(self):
        return (torch.ones(1), 2.0), {}


class SubWithBuiltinInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x - y
        return z

    def get_example_inputs(self):
        return (torch.ones(1).to(torch.int64), 2), {}
