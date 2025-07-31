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


class SimpleArangeStartStepWithInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Add operator is used to test that arange_start_step is fused to const_tensor
        """
        start = 1
        end = 5
        y = torch.arange(start, end)
        z = torch.add(x, y)
        return z

    def get_example_inputs(self):
        return (torch.randn(4),), {}


class SimpleArangeStartStepWithFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Add operator is used to test that arange_start_step is fused to const_tensor
        """
        start = 1.0
        end = 3.0
        step = 0.5
        y = torch.arange(start=start, end=end, step=step)
        z = torch.add(x, y)
        return z

    def get_example_inputs(self):
        return (torch.randn(4),), {}


class SimpleArangeStartStepWithDifferentType(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Add operator is used to test that arange_start_step is fused to const_tensor
        """
        start = 1
        end = 5.0
        y = torch.arange(start, end)
        z = torch.add(x, y)
        return z

    def get_example_inputs(self):
        return (torch.randn(4),), {}
