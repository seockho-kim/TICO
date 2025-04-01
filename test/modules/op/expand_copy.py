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


class SimpleExpandCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.expand_copy(x, size=(1, 3, 4))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 4),)


class SimpleExpandCopyMinusDim(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.expand_copy(x, size=(3, -1))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 4),)
