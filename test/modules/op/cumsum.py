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


class SimpleCumsumDim0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        result = torch.cumsum(x, dim)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 5), 0)


class SimpleCumsumInt32(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        result = torch.cumsum(x, dim)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 5).to(torch.int32), 0)


class SimpleCumsumInt64(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        result = torch.cumsum(x, dim)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 5).to(torch.int64), 0)


class SimpleCumsumDim1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        result = torch.cumsum(x, dim)
        return result

    def get_example_inputs(self):
        return (torch.randn(1, 5), 1)


class SimpleCumsumDim2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        result = torch.cumsum(x, dim)
        return result

    def get_example_inputs(self):
        return (torch.randn(4, 3, 5), 2)
