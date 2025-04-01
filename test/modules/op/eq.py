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


class SimpleEq(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = torch.eq(x, y)
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3), torch.randn(3, 3))


class SimpleEqWithEqualSign(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x == y
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3), torch.randn(3, 3))


class SimpleEqWithScalarInt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x == y
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3).to(torch.int64), 2)


class SimpleEqWithScalarFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x == y
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3), 2.0)


class SimpleEqWithDifferentTypeScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x == y
        return z

    def get_example_inputs(self):
        return (torch.randn(3, 3), 2)
