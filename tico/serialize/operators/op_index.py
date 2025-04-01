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

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import IndexArgs


@register_node_visitor
class IndexTensorVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.index.Tensor]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        args = IndexArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        tensor = args.input
        indices = args.indices

        # TODO Support multiple indices
        if len(indices) - indices.count(None) > 1:  # type: ignore[arg-type]
            raise NotYetSupportedError(
                "Multiple indices is not supported yet in aten.index.Tensor"
            )

        # find the lonely index
        # ex. indices = [None, tensor, None] # index: tensor, axis: 1
        # ex. indices = [1] # index: 1, axis 0
        # ex. indices = [tensor] # index: tensor, axis 0
        index = None
        axis = None
        for axis_, index_ in enumerate(indices):
            if index_ is not None:
                index = index_  # type: ignore[assignment]
                axis = axis_  # type: ignore[assignment]
                break

        assert index is not None, index
        assert axis is not None, axis

        inputs = [tensor, index]
        outputs = [node]

        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.GATHER, self._op_codes
        )
        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.GatherOptions
        option = circle.GatherOptions.GatherOptionsT()
        option.axis = axis  # type: ignore[assignment]
        # TODO option.batchDims
        operator.builtinOptions = option

        return operator
