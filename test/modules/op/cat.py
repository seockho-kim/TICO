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


class SimpleCatWithDim(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        z = torch.cat(tensors=tensors, dim=1)
        return z

    def get_example_inputs(self):
        return ((torch.zeros(3, 3), torch.ones(3, 3)),), {}


class SimpleCatDefault(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        z = torch.cat(tensors)
        return z

    def get_example_inputs(self):
        return ((torch.zeros(3), torch.ones(2)),), {}


class SimpleCatThreeTensors(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        z = torch.cat(tensors=tensors, dim=2)
        return z

    def get_example_inputs(self):
        return (
            (
                torch.zeros(3, 3, 2),
                torch.ones(3, 3, 1),
                torch.ones(3, 3, 1),
            ),
        ), {}
