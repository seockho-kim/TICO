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
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
)

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class FakeQuantizeTensorQParamPerTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        s = torch.ones([1])
        zp = torch.zeros([1])
        qmin = 0
        qmax = 255

        return torch.fake_quantize_per_tensor_affine(input, s, zp, qmin, qmax)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 32, 32),)


class DecomposeFakeQuantizeTensorQParamPerTensor(SinglePassValueTest):
    def test_pass(self):
        self.setup(FakeQuantizeTensorQParamPerTensor())
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams],
            ),
            1,
        )

        self.run_value_test(DecomposeFakeQuantizeTensorQParams())
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams],
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.quantized_decomposed.quantize_per_tensor.default],
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.quantized_decomposed.dequantize_per_tensor.default],
            ),
            1,
        )
