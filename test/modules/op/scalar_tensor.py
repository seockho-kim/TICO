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


class SimpleScalarTensor(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.scalar_tensor(2.0)
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.randn(3),), {}


class SimpleScalarTensorInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.scalar_tensor(2)
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.randn(3),), {}


class SimpleScalarTensorBool(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.scalar_tensor(True)
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.randn(3),), {}
