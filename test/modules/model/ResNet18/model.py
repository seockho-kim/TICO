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
from torchvision.models.resnet import resnet18, ResNet18_Weights

from test.modules.base import TestModuleBase


class ResNet18(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT).to("cpu")

    def forward(self, x):
        return self.model(x)

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 3, 16, 16),), {}
