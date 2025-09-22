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
import torch.nn.functional as F

from test.modules.base import TestModuleBase


class SimpleConstantPad1DInput4D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # Pad the last dimension (left/right)
        result = F.pad(tensor, [2, 2], mode="constant", value=0)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}


class SimpleConstantPad2DInput4D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # Pad the last two dimensions (left/right/top/bottom)
        result = F.pad(tensor, [0, 1, 0, 1], mode="constant", value=0)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 2, 3, 3),), {}


class SimpleConstantPad3DInput4D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # Pad the last three
        result = F.pad(tensor, [1, 2, 3, 4, 5, 6], mode="constant", value=0)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}


class SimpleConstantPad4DInput4D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # Pad the last three
        result = F.pad(tensor, [1, 2, 3, 4, 5, 6, 7, 8], mode="constant", value=0)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}


class SimpleConstantPad3DInput3D(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        # Pad the last three
        result = F.pad(tensor, [1, 2, 3, 4, 5, 6], mode="constant", value=0)
        return result

    def get_example_inputs(self):
        return (torch.randn(2, 3, 3),), {}


class DifferentRightBottom(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        result = F.pad(tensor, [0, 2, 0, 4], mode="constant", value=0)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}


class ConstantPad2d(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d((0, 2, 0, 4), value=0)

    def forward(self, tensor):
        return self.pad(tensor)

    def get_example_inputs(self):
        return (torch.randn(1, 2, 3, 3),), {}


class ConstantPad2dWith3dInput(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ConstantPad2d((0, 2, 0, 4), value=0)

    def forward(self, tensor):
        return self.pad(tensor)

    def get_example_inputs(self):
        return (torch.randn(2, 3, 3),), {}
