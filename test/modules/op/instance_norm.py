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

from test.utils.tag import skip


class SimpleInstanceNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 20
        self.C = 6
        self.H = 10
        self.W = 10
        self.num_features = self.C

    def forward(self, x, weight, bias):
        x = torch.nn.functional.instance_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=weight,
            bias=bias,
            use_input_stats=True,
            momentum=0.1,
            eps=1e-05,
        )
        # x = torch.ops.aten.instance_norm(x, weight, bias, running_mean = None, running_var = None, momentum=0.1, eps=1e-5, use_input_stats = True, cudnn_enabled=False)
        return (x,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.H, self.W)
        weight = torch.randn(self.C)
        bias = torch.randn(self.C)
        return (
            tensor,
            weight,
            bias,
        )


class SimpleInstanceNorm2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.N = 20
        self.C = 6
        self.H = 10
        self.W = 10

        # x = torch.ops.aten.instance_norm(x, p_instnorm_weight, p_instnorm_weight, running_mean = None, running_var = None, momentum=0.1, eps=1e-5, use_input_stats = True, cudnn_enabled=True)
        # input_specs=[InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_instnorm_weight'), target='instnorm.weight', persistent=None), InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='p_instnorm_bias'), target='instnorm.bias', persistent=None), ...
        self.instnorm = torch.nn.InstanceNorm2d(num_features=self.C, affine=True)

    def forward(self, x):
        x = self.instnorm(x)
        return (x,)

    def get_example_inputs(self):
        tensor = torch.randn(self.N, self.C, self.H, self.W)
        return (tensor,)
