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
from torch.nn import functional as F

from test.modules.base import TestModuleBase

from test.utils.tag import test_without_inference


class SimpleLinear(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg):
        return self.linear(arg)

    def get_example_inputs(self):
        return (torch.randn(3, 3),), {}

    # TODO enable this after introducing onert in CI.
    # def get_dynamic_shapes(self):
    #     return {"arg": {0: Dim("batch")}}


class LinearWithDictOutput(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        return {"x": x, "x_minus_1": x - 1}

    def get_example_inputs(self):
        return (torch.randn(2, 10),), {}


class LinearWithTreeOutput(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        return {"x": x, "x_varients": [x - 1, x + 1]}

    def get_example_inputs(self):
        return (torch.randn(2, 10),), {}


class LinearWithUnusedInput(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, arg, attn_mask):
        return self.linear(arg)

    def get_example_inputs(self):
        return (torch.randn(3, 3), None), {}


@test_without_inference
class FQLinearWithFp32Bias(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3, 3))
        self.bias = torch.nn.Parameter(torch.ones(3))

    def forward(self, inp):
        scale = torch.ones(3)
        zero_point = torch.zeros(3)
        axis = 0
        qmin = -32768
        qmax = 32767
        quant_inp = torch.fake_quantize_per_tensor_affine(inp, 1.0, 0, qmin, qmax)
        quant_weight = torch.fake_quantize_per_channel_affine(
            self.weight, scale, zero_point, axis, qmin, qmax
        )
        output = F.linear(quant_inp, quant_weight, bias=self.bias)
        output = torch.fake_quantize_per_tensor_affine(output, 1.0, 0, qmin, qmax)

        return output

    def get_example_inputs(self):
        return (torch.randn(3, 3),), {}
