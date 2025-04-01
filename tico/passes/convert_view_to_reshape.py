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

from tico.passes import ops
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import ViewArgs


@trace_graph_diff_on_pass
class ConvertViewToReshape(PassBase):
    """
    The latest Torch version converts `torch.view` to `torch.ops.aten.reshape`.
    It is necessary to align with Torch Dynamo and provide reshape-centric optimization passes.
    Converting views to reshapes is considered more secure compared to the reverse process.
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

            if node.target not in ops.aten.view:
                continue

            _ = ViewArgs(*node.args, **node.kwargs)

            with graph.inserting_after(node):
                reshape_node = graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=node.args,
                    kwargs=node.kwargs,
                )

            node.replace_all_uses_with(reshape_node, propagate_meta=True)

            modified = True
            logger.debug(f"{node.name} is replaced with {reshape_node.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
