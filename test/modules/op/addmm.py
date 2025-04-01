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


class SimpleAddmmWith1DInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, lhs, rhs):
        out = torch.addmm(input, lhs, rhs, beta=1.0, alpha=2.0)
        return out

    def get_example_inputs(self):
        return (
            torch.randn(5),
            torch.randn(3, 4),
            torch.randn(4, 5),
        )


class SimpleAddmmWith2DInput(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, lhs, rhs):
        out = torch.addmm(input, lhs, rhs, beta=3.0, alpha=2.0)
        return out

    def get_example_inputs(self):
        return (
            torch.randn(3, 5),
            torch.randn(3, 4),
            torch.randn(4, 5),
        )


class SimpleAddmmWithZeroBeta(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, lhs, rhs):
        out = torch.addmm(input, lhs, rhs, beta=0, alpha=2.0)
        return out

    def get_example_inputs(self):
        return (
            torch.randn(3, 5),
            torch.randn(3, 4),
            torch.randn(4, 5),
        )


class SimpleAddmmWithZeroAlpha(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, lhs, rhs):
        out = torch.addmm(input, lhs, rhs, beta=1.0, alpha=0)
        return out

    def get_example_inputs(self):
        return (
            torch.randn(3, 5),
            torch.randn(3, 4),
            torch.randn(4, 5),
        )


class SimpleAddmmWithZeroAlphaAndBeta(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, lhs, rhs):
        out = torch.addmm(input, lhs, rhs, beta=0, alpha=0)
        return out

    def get_example_inputs(self):
        return (
            torch.randn(3, 5),
            torch.randn(3, 4),
            torch.randn(4, 5),
        )


class SimpleAddmmWithNanInputZeroBeta(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, lhs, rhs):
        out = torch.addmm(input, lhs, rhs, beta=0, alpha=1.0)
        return out

    def get_example_inputs(self):
        return (
            torch.tensor([1, float("nan")]),
            torch.randn(1, 4),
            torch.randn(4, 2),
        )
