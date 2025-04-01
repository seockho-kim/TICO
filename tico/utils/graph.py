# Portions of this file are adapted from code originally authored by
# Meta Platforms, Inc. and affiliates, licensed under the BSD-style
# license found in the LICENSE file in the root directory of their source tree.

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

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram
from torch.export.exported_program import InputKind, InputSpec, TensorArgument

from tico.utils.utils import get_fake_mode


def is_torch_param(node: torch.fx.Node, ep: ExportedProgram):
    assert node.op == "placeholder"

    return node.name in ep.graph_signature.inputs_to_parameters


def is_torch_buffer(node: torch.fx.Node, ep: ExportedProgram):
    assert node.op == "placeholder"

    return node.name in ep.graph_signature.inputs_to_buffers


def get_torch_param_value(node: torch.fx.Node, ep: ExportedProgram):
    assert isinstance(node, torch.fx.Node)
    assert node.op == "placeholder"
    assert (
        node.name in ep.graph_signature.inputs_to_parameters
    ), "Node {node.name} is not in the parameters"  # FIX CALLER UNLESS

    param_name = ep.graph_signature.inputs_to_parameters[node.name]
    named_params = dict(ep.named_parameters())
    assert param_name in named_params

    return named_params[param_name].data


def get_torch_buffer_value(node: torch.fx.Node, ep: ExportedProgram):
    assert isinstance(node, torch.fx.Node)
    assert node.op == "placeholder"
    assert (
        node.name in ep.graph_signature.inputs_to_buffers
    ), "Node {node.name} is not in the buffers"  # FIX CALLER UNLESS

    buf_name = ep.graph_signature.inputs_to_buffers[node.name]
    named_buf = dict(ep.named_buffers())
    assert buf_name in named_buf

    return named_buf[buf_name]


def get_first_user_input(exported_program: ExportedProgram) -> Optional[torch.fx.Node]:
    """Returns the first user input node in the graph."""
    first_user_input: Optional[torch.fx.Node] = None
    graph_module = exported_program.graph_module
    graph: torch.fx.Graph = graph_module.graph
    for node in graph.nodes:
        if (
            node.op == "placeholder"
            and node.name in exported_program.graph_signature.user_inputs
        ):
            first_user_input = node
            break

    return first_user_input


def generate_fqn(prefix: str, exported_program: ExportedProgram):
    """
    Generate fully-qualized name for constants.

    This function prevents `exported_program.constants` from having duplicate keys.
    """
    cnt = len(exported_program.constants)
    while True:
        if f"{prefix}{cnt}" in exported_program.constants:
            cnt += 1
            continue
        break
    return f"{prefix}{cnt}"


def create_input_spec(node, input_kind: InputKind):
    """
    @ref https://pytorch.org/docs/stable/export.ir_spec.html#placeholder
    """
    if input_kind == InputKind.CONSTANT_TENSOR:
        return InputSpec(
            kind=InputKind.CONSTANT_TENSOR,
            arg=TensorArgument(name=node.name),
            target=node.target,  # type: ignore[arg-type]
            persistent=True,
        )
    else:
        raise NotImplementedError("NYI")


def validate_input_specs(exported_program):
    name_to_spec_dict = {
        s.arg.name: s for s in exported_program.graph_signature.input_specs
    }

    for node in exported_program.graph.nodes:
        if node.op != "placeholder":
            continue

        if node.name not in name_to_spec_dict:
            raise RuntimeError(
                "Placeholder node {node.name} does not have corresponding input spec!"
            )


def add_placeholder(
    exported_program: ExportedProgram,
    tensor: torch.Tensor,
    prefix: str,
) -> torch.fx.Node:
    """
    Add a placeholder to the graph and update the exported program.
    """
    fqn_name = generate_fqn(prefix, exported_program)

    # Get fake mode before adding placeholder
    fake_mode = get_fake_mode(exported_program)

    first_user_input = get_first_user_input(exported_program)
    if not first_user_input:
        # Placeholder nodes must be the first N nodes in the nodes list of a graph.
        # Therefore, insert the newly created placeholders at the start of the node list.
        assert exported_program.graph.nodes
        first_node = list(exported_program.graph.nodes)[0]
        first_user_input = first_node

    # Add a placeholder to the graph.
    with exported_program.graph.inserting_before(first_user_input):
        const_node = exported_program.graph.placeholder(fqn_name)

    const_node.meta["val"] = fake_mode.from_tensor(tensor, static_shapes=True)
    const_node.meta["val"].constant = tensor

    # Add a new constant to the exported program.
    exported_program.constants[const_node.name] = tensor

    # Use update (instead of append) if this assert is violated
    assert const_node.name not in [
        s.arg.name for s in exported_program.graph_signature.input_specs
    ]

    # Append the new input spec.
    exported_program.graph_signature.input_specs.append(
        create_input_spec(const_node, InputKind.CONSTANT_TENSOR)
    )

    # Get old input specs
    name_to_spec_dict = {
        s.arg.name: s for s in exported_program.graph_signature.input_specs
    }

    # Add the new constants to input specs dict.
    name_to_spec_dict.update(
        {const_node.name: create_input_spec(const_node, InputKind.CONSTANT_TENSOR)}
    )

    # Generate new input spec *in the same order of nodes*
    # IMPORTANT Input specs and their placeholder nodes must have the same order.
    new_input_specs = []
    for node in exported_program.graph.nodes:
        if node.op != "placeholder":
            continue
        new_input_specs.append(name_to_spec_dict[node.name])
    exported_program.graph_signature.input_specs = new_input_specs

    return const_node


def is_single_value_tensor(t: torch.Tensor):
    if len(t.size()) == 0:
        return True
    if len(t.size()) == 1 and t.size()[0] == 1:
        return True

    return False
