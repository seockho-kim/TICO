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


class SimpleGelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()

    def forward(self, tensor):
        result = self.gelu(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(3, 3),)


class GeluWithApproximate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU(approximate="tanh")

    def forward(self, tensor):
        result = self.gelu(tensor)
        return result

    def get_example_inputs(self):
        return (torch.randn(3, 3),)
