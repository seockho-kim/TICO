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


class SimpleClone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        # Intentionally insert `add` op as there's no op after removing no-op.
        ret = torch.clone(input_)
        ret += 2.0
        return ret

    def get_example_inputs(self):
        return (torch.randn(3, 5, 4),)


class SimpleCloneWithMemoryFormat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src: torch.Tensor):
        dst = src.clone(memory_format=torch.preserve_format)
        # Intentionally insert `add` op as there's no op after removing no-op.
        dst += 2
        return dst

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4),)


class SimpleCloneWithMemoryFormatContiguous(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src: torch.Tensor):
        dst = src.clone(memory_format=torch.contiguous_format)
        # Intentionally insert `add` op as there's no op after removing no-op.
        dst += 2
        return dst

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4),)


class SimpleCloneWithMemoryFormatChannelsLast(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src: torch.Tensor):
        dst = src.clone(memory_format=torch.contiguous_format)
        # Intentionally insert `add` op as there's no op after removing no-op.
        dst += 2
        return dst

    def get_example_inputs(self):
        return (torch.randn(1, 3, 5, 4).to(memory_format=torch.channels_last),)


# TODO Add negative test
# dst = src.clone(memory_format=torch.channels_last) -> NOT SUPPORTED YET
