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

import unittest

import torch
from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import PT2EConfig
from tico.experimental.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.serialize.quant_param import QPARAM_KEY

from test.modules.op.sub import SimpleSub
from test.utils.helper import num_of_ops


class FoldQuantOpsTest(unittest.TestCase):
    def test_pass(self):
        m: SimpleSub | torch.nn.Module = SimpleSub().eval()
        assert isinstance(m, SimpleSub)
        example_inputs = m.get_example_inputs()

        q_m = prepare(m, PT2EConfig(), args=example_inputs)

        # Calibration
        for _ in range(10):
            cal_in = m.get_example_inputs()
            q_m(*cal_in)

        q_m = convert(q_m)

        ep = torch.export.export(q_m, example_inputs)

        # input, other, sub
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            3,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            3,
        )
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.sub.Tensor:
                self.assertFalse(QPARAM_KEY in node.meta)

        target_pass = FoldQuantOps()
        target_pass.call(ep)

        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            0,
        )
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.sub.Tensor:
                self.assertTrue(QPARAM_KEY in node.meta)
