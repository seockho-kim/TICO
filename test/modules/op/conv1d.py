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


class Conv1dPaddingZero(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class Conv1dPaddingOne(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class Conv1dPaddingTwo(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=2
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class Conv1dPaddingValid(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding="valid"
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class Conv1dPaddingSame(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding="same"
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class Conv1dNoBias(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(10, 3, 128),), {}


class DepthwiseConv1d(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128
        )

    def forward(self, input):
        return self.conv1d(input)

    def get_example_inputs(self):
        return (torch.randn(1, 128, 6),), {}
