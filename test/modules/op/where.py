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


class SimpleWhereWithTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, condition, result_true, result_false):
        result = torch.where(condition, result_true, result_false)  # type: ignore[arg-type]
        return result

    def get_example_inputs(self):
        condition = torch.empty(3, 3).uniform_(0, 1)
        condition = torch.bernoulli(condition).bool()

        return (
            condition,
            torch.randint(0, 10, (3, 3), dtype=torch.float32),
            torch.randint(0, 10, (3, 3), dtype=torch.float32),
        )


class SimpleWhereWithScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, condition, result_true, result_false):
        result = torch.where(condition, result_true, result_false)  # type: ignore[arg-type]
        return result

    def get_example_inputs(self):
        condition = torch.empty(3, 3).uniform_(0, 1)
        condition = torch.bernoulli(condition).bool()

        return (
            condition,
            1.0,
            0.0,
        )


class SimpleWhereWithConstantTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        result_true = torch.tensor([1, 2, 3], dtype=torch.int32)
        result_false = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
        self.register_buffer("result_true_const", result_true)
        self.register_buffer("result_false_const", result_false)

    def forward(self, condition):
        result = torch.where(
            condition, input=self.result_true_const, other=self.result_false_const  # type: ignore[arg-type]
        )
        return result

    def get_example_inputs(self):
        condition = torch.empty(3).uniform_(0, 1)
        condition = torch.bernoulli(condition).bool()

        return (condition,)
