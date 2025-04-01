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

from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.validate_args_kwargs import EmbeddingArgs


@register_node_visitor
class EmbeddingVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten.embedding.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.GATHER, self._op_codes
        )
        args = EmbeddingArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        # do not need to handle optional parameters, since all options are related to Embedding Table values.
        embedding_table = args.weight
        index_tensor = args.indices

        inputs = [embedding_table, index_tensor]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        # Op-specific option
        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.GatherOptions
        option = circle.GatherOptions.GatherOptionsT()
        option.axis = 0
        operator.builtinOptions = option

        # Op-specific option does not exists for Embedding.
        return operator
