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
from circle_schema import circle
from tico.serialize.operators import (  # DO NOT REMOVE! (required to build _node_visitor_dict)
    op_mm,
)

from test.modules.op.mm import SimpleMatmul
from test.unit_test.serialize_test.operator.fixture import SingleOpGraphFixture


class MatmulVisitorTest(unittest.TestCase):
    def test_op_mm_to_fullyconnected(self):
        bp = SingleOpGraphFixture(SimpleMatmul(), torch.ops.aten.mm.default)
        mmVisitor = bp.target_visitor()
        res = mmVisitor.define_node(bp.target_node(), prior_latency=False)

        self.assertTrue(
            isinstance(
                res.builtinOptions, circle.FullyConnectedOptions.FullyConnectedOptionsT
            )
        )

    def test_op_mm_to_bmm(self):
        bp = SingleOpGraphFixture(SimpleMatmul(), torch.ops.aten.mm.default)
        mmVisitor = bp.target_visitor()
        res = mmVisitor.define_node(bp.target_node(), prior_latency=True)

        self.assertTrue(
            isinstance(
                res.builtinOptions, circle.BatchMatMulOptions.BatchMatMulOptionsT
            )
        )
