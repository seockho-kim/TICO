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

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass


@trace_graph_diff_on_pass
class ConvertSymSizeToCircleShape(PassBase):
    """
    This pass converts torch.ops.aten.sym_size.int operations to circle_custom::shape.

    The circle_custom::shape operator allows preserving dynamic shape information
    in the Circle model. This is essential for models with dynamic batch sizes or other dynamic dimensions.

    Example:
        Before: %sym_size_int_1 = call_function[target=torch.ops.aten.sym_size.int](args=(%x, 0))
        After:  %shape_0 = call_function[target=torch.ops.circle_custom.shape](args=(%x,))
                %slice_0 = call_function[target=torch.ops.aten.slice.Tensor](args=(%shape_0, 0, 0, 1, 1))
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            if node.target == torch.ops.aten.sym_size.int:
                # sym_size.int has args: (input, dim)
                input_tensor = node.args[0]
                dim = node.args[1]

                # Create circle_custom::shape node
                with graph.inserting_after(node):
                    shape_node = create_node(
                        graph,
                        torch.ops.circle_custom.shape,
                        args=(input_tensor,),
                    )

                # Set metadata for shape_node
                if "val" in input_tensor.meta:
                    input_val = input_tensor.meta["val"]
                    rank = len(input_val.shape)
                    # shape output is a 1D tensor of size rank, dtype int32
                    # We use a real tensor here as a placeholder for metadata
                    shape_node.meta["val"] = torch.zeros(rank, dtype=torch.int32)

                # Extract the specific dimension using slice
                with graph.inserting_after(shape_node):
                    slice_node = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        args=(shape_node, 0, dim, dim + 1, 1),
                    )
                    # slice output is 1D tensor of size 1
                    slice_node.meta["val"] = torch.zeros(1, dtype=torch.int32)

                # Replace all uses
                node.replace_all_uses_with(slice_node, propagate_meta=False)
                modified = True

                logger.debug(
                    f"Converted {node.name} (sym_size.int) to {shape_node.name} (circle_custom::shape) + {slice_node.name} (slice)"
                )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
