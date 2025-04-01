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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass


@trace_graph_diff_on_pass
class ConvertIndexToResizeNearestNeighbor(PassBase):
    """
    This pass converts `aten.index` to `circle_custom.resize_nearest_neighbor` when it is possible.

    `torch.nn.functional.interpolate` is converted to `aten.index` op.

    [EXAMPLE]
      class InterpolateDouble(torch.nn.Module):
          def __init__(self):
              super().__init__()

          def forward(self, x):
              return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

          def get_example_inputs(self):
              return (torch.randn(1, 2, 3, 4),)

    [EXPORTED GRAPH]
      [constants]
        _prop_tensor_constant0 = tensor([0, 0, 1, 1, 2, 2, 3, 3]
        _prop_tensor_constant1 = tensor([[0], [0], [1], [1], [2], [2]])

      [graph]
        %_prop_tensor_constant0 : [num_users=1] = placeholder[target=_prop_tensor_constant0]
        %_prop_tensor_constant1 : [num_users=1] = placeholder[target=_prop_tensor_constant1]
        %x : [num_users=1] = placeholder[target=x]
        %_to_copy : [num_users=1] = call_function[target=torch.ops.aten._to_copy.default](args = (%x,), kwargs = {dtype: torch.float32})
        %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%_to_copy, [None, None, %_prop_tensor_constant1, %_prop_tensor_constant0]), kwargs = {})
        %_to_copy_3 : [num_users=1] = call_function[target=torch.ops.aten._to_copy.default](args = (%index,), kwargs = {dtype: torch.float32})
        return (_to_copy_3,)

    [BEFORE PASS]
      input - aten.index - output

    [AFTER PASS]
      input - aten.permute(NCHW_to_NHWC) - circle_custom.resize_nearest_neighbor - aten.permute(NHWC_to_NCHW) - output
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            if node.target != torch.ops.aten.index.Tensor:
                continue

            assert len(node.args) == 2, len(node.args)
            input_tensor, indices = node.args
            assert isinstance(input_tensor, torch.fx.Node), type(input_tensor)
            assert isinstance(indices, list)

            # Only support 4-D tensor
            if len(indices) != 4:
                continue
            # indices = [None, None, H index, W index]
            N, C, H, W = indices
            if N != None or C != None:
                continue
            if not isinstance(H, torch.fx.Node):
                continue
            if not isinstance(W, torch.fx.Node):
                continue
            constants_dict = exported_program.constants
            if (H.name not in constants_dict) or (W.name not in constants_dict):
                continue
            H_index, W_index = constants_dict[H.name], constants_dict[W.name]
            input_tensor_shape = extract_shape(input_tensor)
            input_tensor_H, input_tensor_W = (
                input_tensor_shape[2],
                input_tensor_shape[3],
            )
            if H_index.size()[0] % input_tensor_H != 0:
                continue
            scale_factor = int(H_index.size()[0] / input_tensor_H)
            # H and W should be resized with same ratio.
            if scale_factor != W_index.size()[0] / input_tensor_W:
                continue
            expected_H_index = []
            expected_W_index = []
            # Please refer to above `_prop_tensor_constant1` constant in the example.
            for i in range(input_tensor_H):
                expected_H_index += [[i]] * scale_factor
            # Please refer to above `_prop_tensor_constant0` constant in the example.
            for i in range(input_tensor_W):
                expected_W_index += [i] * scale_factor
            if not torch.all(
                torch.eq(H_index, torch.tensor(expected_H_index))
            ) or not torch.all(torch.eq(W_index, torch.tensor(expected_W_index))):
                continue
            expected_shape = [
                input_tensor_shape[0],
                input_tensor_shape[1],
                len(expected_H_index),
                len(expected_W_index),
            ]
            assert expected_shape == list(extract_shape(node))

            with graph.inserting_before(node):
                nchw_to_nhwc = graph.call_function(
                    torch.ops.aten.permute.default, args=(input_tensor, [0, 2, 3, 1])
                )
                resize_nearest_neighbor = graph.call_function(
                    torch.ops.circle_custom.resize_nearest_neighbor,
                    args=(nchw_to_nhwc, [len(expected_H_index), len(expected_W_index)]),
                )
                nhwc_to_nchw = graph.call_function(
                    torch.ops.aten.permute.default,
                    args=(resize_nearest_neighbor, [0, 3, 1, 2]),
                )
                # Not set meta for propagating replacing node's meta.
                node.replace_all_uses_with(nhwc_to_nchw, propagate_meta=True)

            modified = True
            logger.debug(
                f"{node.name} is replaced with {resize_nearest_neighbor.name} operator"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
