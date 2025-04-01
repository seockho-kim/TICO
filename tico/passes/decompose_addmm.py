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
from tico.utils.graph import add_placeholder
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val
from tico.utils.validate_args_kwargs import AddmmArgs


@trace_graph_diff_on_pass
class DecomposeAddmm(PassBase):
    """
    Let's decompose addmm to add + mul + matmul.

    [BEFORE]

    input   mat1    mat2    beta    alpha
    |       |       |       |       |
    --------------addmm--------------
                    |
                    out

    [AFTER]

    input   beta    mat1    mat2    alpha
    |       |       |       |        |
    ---mul---       ---mm----        |
        |               |            |
        |               -----mul-----
        |                     |
        ---------add----------
                  |
                  out

    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        gm = exported_program.graph_module
        graph: torch.fx.Graph = gm.graph
        modified = False

        for node in graph.nodes:
            if node.op != "call_function":
                continue

            if node.target in [
                torch.ops.aten.addmm.default,
            ]:
                args = AddmmArgs(*node.args, **node.kwargs)
                input = args.input
                mat1 = args.mat1
                mat2 = args.mat2
                beta = args.beta
                alpha = args.alpha

                with graph.inserting_before(node):
                    # out = beta * input + alpha * (mat1 @ mat2)
                    matmul = graph.call_function(
                        torch.ops.aten.mm.default, (mat1, mat2)
                    )
                    set_new_meta_val(matmul)

                    if beta == 1:
                        bias: torch.fx.Node | torch.Tensor = input
                    elif beta == 0:
                        bias = add_placeholder(
                            exported_program,
                            torch.zeros(extract_shape(input)),
                            f"{node.name}_beta_zeros",
                        )
                    else:
                        bias = graph.call_function(
                            torch.ops.aten.mul.Tensor, (input, beta)
                        )

                    if alpha == 1:
                        scaled_matmul: torch.fx.Node | torch.Tensor = matmul
                    elif alpha == 0:
                        scaled_matmul = add_placeholder(
                            exported_program,
                            torch.zeros(extract_shape(matmul)),
                            f"{node.name}_alpha_zeros",
                        )
                    else:
                        scaled_matmul = graph.call_function(
                            torch.ops.aten.mul.Tensor, (matmul, alpha)
                        )

                    result = graph.call_function(
                        torch.ops.aten.add.Tensor, (bias, scaled_matmul)
                    )

                node.replace_all_uses_with(result, propagate_meta=True)

                modified = True

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)
