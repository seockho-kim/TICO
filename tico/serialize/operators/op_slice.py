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

import copy
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
from tico.utils.errors import InvalidArgumentError
from tico.utils.validate_args_kwargs import SliceArgs


@register_node_visitor
class SliceCopyVisitor(NodeVisitor):
    """
    NOTE `torch.slice_copy`'s behavior matches with `strided slice` of CIRCLE, not `slice`.
    """

    target: List[torch._ops.OpOverload] = [
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.slice_copy.Tensor,
    ]

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        super().__init__(op_codes, graph)

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.STRIDED_SLICE, self._op_codes
        )

        args = SliceArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        dim = args.dim
        start = args.start
        end = args.end
        step = args.step

        input_tensor: circle.Tensor.TensorT = self.graph.get_tensor(input)
        input_shape: List[int] = input_tensor.shape

        if start is None:
            start = 0
        if end is None:
            end = input_shape[dim]
        if step is None:
            step = 1

        assert dim is not None
        assert (
            -len(input_shape) <= dim < len(input_shape)
        ), "Cannot reach here (Dimension Out of Range error must be thrown by torch)"

        if dim < 0:
            dim = dim % len(input_shape)

        assert isinstance(start, int), type(start)
        assert isinstance(end, int), type(end)
        assert isinstance(step, int), type(step)

        if start < -input_shape[dim]:  # (-inf, -M)
            """
            WHY is 0?
            start = -input_shape[dim] % input_shape[dim]
            """
            start = 0
        elif -input_shape[dim] <= start < 0:  # [-M, 0)
            start %= input_shape[dim]
        elif 0 <= start < input_shape[dim]:  # [0, M)
            start = start
        elif input_shape[dim] <= start:  # [M, +inf)
            start = input_shape[dim]
        else:
            assert False, "Cannot reach here"

        if end < -input_shape[dim]:  # (-inf, -M)
            """
            WHY is 0?
            end = -input_shape[dim] % input_shape[dim]
            """
            end = 0
        elif -input_shape[dim] <= end < 0:  # [-M, 0)
            end %= input_shape[dim]
        elif 0 <= end < input_shape[dim]:  # [0, M)
            end = end
        elif input_shape[dim] <= end:  # [M, +inf)
            end = input_shape[dim]
        else:
            assert False, "Cannot reach here"

        assert 0 <= dim and dim < len(input_shape), dim
        assert 0 <= start and start < input_shape[dim], start
        assert 0 <= end and end <= input_shape[dim], end
        assert 0 < step, "Restriction of torch.slice_copy"

        if end <= start:
            """
            CONSTRAINTS
            In torch, 'end <= start' condition generates zero tensor with a peculiar shape - ex. tensor([], size=(5,0,5))
            In circle, it's not accepted at all.
            """
            raise InvalidArgumentError(
                f"end({end}) must be greater than start ({start})"
            )

        # Build new arguments
        rank = len(input_shape)

        begin_shape = [0] * rank
        begin_shape[dim] = start
        begin_shape_tensor = torch.as_tensor(begin_shape, dtype=torch.int32)

        end_shape = copy.deepcopy(input_shape)
        end_shape[dim] = end
        end_shape_tensor = torch.as_tensor(end_shape, dtype=torch.int32)

        stride_shape = [1] * rank
        stride_shape[dim] = step
        stride_shape_tensor = torch.as_tensor(stride_shape, dtype=torch.int32)

        inputs = [input, begin_shape_tensor, end_shape_tensor, stride_shape_tensor]
        outputs = [node]

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = (
            circle.BuiltinOptions.BuiltinOptions.StridedSliceOptions
        )

        option = circle.StridedSliceOptions.StridedSliceOptionsT()

        operator.builtinOptions = option
        return operator
