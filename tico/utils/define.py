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

from typing import Dict, List

from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.utils import create_builtin_operator, get_op_index


def define_pad_node(
    graph: CircleSubgraph, op_codes: Dict[OpCode, int], inputs: List, outputs: List
) -> circle.Operator.OperatorT:
    def set_pad_option(operator: circle.Operator.OperatorT):
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.PadOptions
        option = circle.PadOptions.PadOptionsT()
        operator.builtinOptions = option

    pad_op_index = get_op_index(circle.BuiltinOperator.BuiltinOperator.PAD, op_codes)
    operator = create_builtin_operator(graph, pad_op_index, inputs, outputs)
    set_pad_option(operator)
    return operator
