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

from test.utils import tag


class SimpleAtenIndexTensorAxis1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([1, 2, 3, 4, 5]))

    def forward(self, x):
        y = x[:, self.idx]
        return y

    def get_example_inputs(self):
        return (torch.randn(3, 20, 4, 5),)


class SimpleAtenIndexTensorAxis2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("idx", torch.tensor([1, 2, 3, 4, 5]))

    def forward(self, x):
        y = x[:, :, self.idx]
        return y

    def get_example_inputs(self):
        return (torch.randn(3, 1, 2048, 5),)


class SimpleIndexTensorBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        idx = torch.tensor([0, 1])
        self.register_buffer("idx", idx)

    def forward(self, x, y):
        z = x + y
        return z[self.idx]

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3))


# Without buffer, there is `aten_clone_default` op which is converted to `CircleTranspose`.
# But, circle-interpreter doesn't support the `CircleTranspose` whose type is int64.
@tag.skip(reason="Not Support Operator")
class SimpleIndexTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        idx = torch.tensor([0, 1])
        z = x + y
        return z[idx]

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3))


class IndexTensor2x1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        idx = torch.tensor([[0], [0]])
        self.register_buffer("idx", idx)

    def forward(self, x, y):
        z = x + y
        return z[self.idx]

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3))


# circle-interpreter doesn't support the `CircleTranspose` whose type is int64.
@tag.skip(reason="Not Support Operator")
class IndexTensorAxis1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z[1, [0, 0]]

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3))
