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

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_mapping import extract_shape
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import InvalidArgumentError
from tico.utils.validate_args_kwargs import ConstantPadNdArgs


@register_node_visitor
class ConstantPadNdVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.constant_pad_nd.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = ConstantPadNdArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input_ = args.input
        pad = args.pad
        val = args.value

        if val != 0:
            raise InvalidArgumentError("Only support 0 value padding.")

        input_shape_len = len(extract_shape(input_))
        padding_size = [[pad[2], pad[3]], [pad[0], pad[1]]]
        if input_shape_len == 3:
            padding_size = [[0, 0]] + padding_size
        elif input_shape_len == 4:
            padding_size = [[0, 0], [0, 0]] + padding_size
        else:
            raise InvalidArgumentError("Only support 3D/4D inputs.")

        paddings = torch.tensor(padding_size, dtype=torch.int32)
        inputs = [input_, paddings]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.PAD, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.PadOptions
        option = circle.PadOptions.PadOptionsT()
        operator.builtinOptions = option

        return operator
