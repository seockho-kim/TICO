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


class SimpleScatterCopyDim0(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Equivalent to .. (y, dim=0, start=6, end=None, step=1)
        z = x.slice_scatter(y, start=6)
        return z

    def get_example_inputs(self):
        return (
            torch.zeros(8, 8),
            torch.ones(2, 8),
        ), {}


class SimpleScatterCopyDim1(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x.slice_scatter(y, dim=1, start=2, end=4, step=1)
        return z

    def get_example_inputs(self):
        return (
            torch.zeros(8, 8),
            torch.ones(8, 2),
        ), {}


class SimpleScatterCopyDim2(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Equivalent to .. (y, dim=2, start=None, end=5, step=1)
        z = x.slice_scatter(y, dim=2, end=5)
        return z

    def get_example_inputs(self):
        return (
            torch.zeros(2, 3, 8),
            torch.ones(2, 3, 5),
        ), {}
