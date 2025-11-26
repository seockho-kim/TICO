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


class SimpleCopy(TestModuleBase):
    """
    Test case: Same shape copy (should be folded away by ConvertCopyToReshape pass)
    5x5 -> 5x5
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(5, 5)), {}


class SimpleCopyWithBroadcastTo(TestModuleBase):
    """
    Test case: Broadcast from 1x5 to 5x5
    This tests the expand + reshape path in ConvertCopyToReshape pass
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(1, 5)), {}


class CopyWithScalarBroadcast(TestModuleBase):
    """
    Test case: Broadcast from 1x1 to 3x3 (scalar-like broadcast)
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(3, 3), torch.randn(1, 1)), {}


class CopyWithRowBroadcast(TestModuleBase):
    """
    Test case: Broadcast from 1x4 to 3x4 (row broadcast)
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(3, 4), torch.randn(1, 4)), {}


class CopyWithColumnBroadcast(TestModuleBase):
    """
    Test case: Broadcast from 3x1 to 3x4 (column broadcast)
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(3, 4), torch.randn(3, 1)), {}


class CopyWith3DTensor(TestModuleBase):
    """
    Test case: 3D tensor copy with same shape
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4), torch.randn(2, 3, 4)), {}


class CopyWith3DBroadcast(TestModuleBase):
    """
    Test case: 3D tensor broadcast from 1x3x4 to 2x3x4
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4), torch.randn(1, 3, 4)), {}


class CopyWithMultiDimBroadcast(TestModuleBase):
    """
    Test case: Multi-dimensional broadcast from 1x1x4 to 2x3x4
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4), torch.randn(1, 1, 4)), {}


class CopyWith4DTensor(TestModuleBase):
    """
    Test case: 4D tensor copy (batch, channel, height, width)
    Broadcast from 1x3x1x1 to 2x3x4x4
    """

    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4, 4), torch.randn(1, 3, 1, 1)), {}
