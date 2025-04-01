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

from tico.serialize.circle_mapping import (
    extract_circle_dtype,
    extract_torch_dtype,
    to_circle_dtype,
)
from tico.serialize.operators.hashable_opcode import OpCode
from tico.serialize.operators.node_visitor import NodeVisitor, register_node_visitor
from tico.serialize.operators.utils import create_builtin_operator, get_op_index
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import ToCopyArgs


@register_node_visitor
class ToCopyVisitor(NodeVisitor):
    target: List[torch._ops.OpOverload] = [torch.ops.aten._to_copy.default]

    def __init__(self, op_codes: Dict[OpCode, int], graph):
        super().__init__(op_codes, graph)

    def define_cast_node(
        self,
        inputs: List[torch.fx.Node],
        outputs: List[torch.fx.Node],
        in_type: int,
        out_type: int,
    ):
        op_index = get_op_index(
            circle.BuiltinOperator.BuiltinOperator.CAST, self._op_codes
        )

        operator = create_builtin_operator(self.graph, op_index, inputs, outputs)

        operator.builtinOptionsType = circle.BuiltinOptions.BuiltinOptions.CastOptions
        option = circle.CastOptions.CastOptionsT()
        option.inDataType = in_type
        option.outDataType = out_type
        operator.builtinOptions = option

        return operator

    def define_node(
        self,
        node: torch.fx.Node,
    ) -> circle.Operator.OperatorT:
        supported_kwargs = ["dtype", "device", "layout"]
        if not all(k in supported_kwargs for k in node.kwargs):
            unsupported_node_kargs = list(node.kwargs.keys())
            for supported_key in supported_kwargs:
                if supported_key in node.kwargs:
                    unsupported_node_kargs.remove(supported_key)
            raise NotYetSupportedError(
                f"Support only {supported_kwargs} kwargs now. Do not support {unsupported_node_kargs}"
            )

        args = ToCopyArgs(*node.args, **node.kwargs)  # type: ignore[arg-type, call-arg]
        input = args.input
        dtype = args.dtype

        input_meta = input.meta["val"]
        # https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout
        # layout is two types: torch.strided(dense Tensors), torch.sparse_coo(sparse COO Tensors)
        if "layout" in input.kwargs and input.kwargs["layout"] != input_meta:
            raise NotYetSupportedError(
                f"Only support when node and its input have same layout: (input layout: {input_meta}), (node layout: {node.kwargs['layout']})."
            )

        if dtype is not None:
            target_type = node.kwargs["dtype"]
        else:
            # device and layout are meaningless
            target_type = extract_torch_dtype(node)
        assert isinstance(target_type, torch.dtype), type(target_type)

        # define cast node
        in_type: int = extract_circle_dtype(input)
        out_type: int = to_circle_dtype(target_type)
        inputs = [input]
        outputs = [node]
        operator = self.define_cast_node(inputs, outputs, in_type, out_type)

        # TODO Support layout conversion

        return operator
