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

# TODO Add a test for `aten.mean` when the operator is supported.


class SimpleMean(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=1)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(4, 4),)


class SimpleMeanKeepDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=1, keepdim=True)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5),)


class SimpleMeanTwoDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=[1, 2])
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5, 5),)


class SimpleMeanTwoDim2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=[2, 1])
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5, 5),)


class SimpleMeanNegativeTwoDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=[-2, -3])
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5, 5),)


class SimpleMeanNegativeTwoDim2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=[-3, -2])
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5, 5),)


class MeanWithRedundantView(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=1, keepdim=True)
        z = torch.ops.aten.view(z, z.shape)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5),)


class MeanWithRedundantViewAndDtype(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.mean(x, dim=1, keepdim=True, dtype=torch.float32)
        z = torch.ops.aten.view(z, z.shape)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(5, 5),)
