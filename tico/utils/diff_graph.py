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

from copy import deepcopy
from difflib import ndiff
from functools import reduce
from logging import DEBUG
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch

from tico.utils.logging import getLogger, LOG_LEVEL


def strdiff(a: str, b: str):
    """
    Get difference in two strings as if linux `diff` command does
    """
    assert isinstance(a, str), f"{a} must be str, type: {type(a)}"
    assert isinstance(b, str), f"{b} must be str, type: {type(b)}"

    changed = []
    for line in ndiff(a.splitlines(keepends=True), b.splitlines(keepends=True)):
        if line.startswith(("-", "+")):
            changed.append(line)
    return "".join(changed)


def disable_when(predicate):
    """
    Disable function only if predicate is true
    """

    def _inner_disable_when(func):
        if predicate:

            def nop(*args, **kwargs):
                pass

            return nop
        else:
            return func

    return _inner_disable_when


LOGGER_THRESHOLD = DEBUG
graph_captured: Optional[str | torch.fx.Graph] = None
const_size_captured: Optional[int] = None


def get_const_size(ep: torch.export.ExportedProgram) -> int:
    """
    Return const tensor's size in **byte**
    """

    def const_size(items):
        const_sum = 0
        for _, tensor in items:
            if len(tensor.size()) == 0:
                # scalar tensor
                const_sum += tensor.dtype.itemsize
            else:
                const_sum += (
                    reduce(lambda x, y: x * y, list(tensor.size()))
                    * tensor.dtype.itemsize
                )
        return const_sum

    constant_tensor_sum = 0

    constant_tensor_sum += const_size(ep.state_dict.items())
    constant_tensor_sum += const_size(ep.constants.items())

    return constant_tensor_sum


@disable_when(LOG_LEVEL > LOGGER_THRESHOLD)
def capture_const(ep: torch.export.ExportedProgram):
    assert isinstance(ep, torch.export.ExportedProgram)

    global const_size_captured
    const_size_captured = get_const_size(ep)


@disable_when(LOG_LEVEL > LOGGER_THRESHOLD)
def log_const(ep: torch.export.ExportedProgram, title: str, recapture: bool):
    assert isinstance(ep, torch.export.ExportedProgram)

    global const_size_captured
    assert const_size_captured is not None
    const_size = get_const_size(ep)
    const_size_diff = const_size - const_size_captured

    # print differences
    logger = getLogger(__name__)
    prefix = f"[{title}]" if title else ""
    if const_size_diff > 0:
        const_size_inc_dec = "has changed (increased)"
    elif const_size_diff == 0:
        const_size_inc_dec = "has unchanged"
    else:
        const_size_inc_dec = "has changed (decreased)"

    percentage_avg_str = ""
    if const_size + const_size_captured == 0:
        percentage_avg_str = "N/A"
    else:
        percentage_avg = (
            float(const_size_diff) / float(const_size + const_size_captured) * 100
        )
        if percentage_avg > 0:
            percentage_avg_str = f"+{percentage_avg:.2f}%"
        else:
            percentage_avg_str = f"{percentage_avg:.2f}%"

    if const_size_diff:
        logger.debug(
            f"{prefix} Total const size {const_size_inc_dec} by {const_size_diff} Bytes"
        )
        logger.debug(f"{const_size_captured}B -> {const_size}B ({percentage_avg_str})")

    if recapture:
        const_size_captured = const_size


@disable_when(LOG_LEVEL > LOGGER_THRESHOLD)
def capture(graph: torch.fx.Graph):
    """
    Capture the start-point graph for graph-diff.
    String diff lines will be printed to debug logger if enabled.

    Args:
        graph (torch.fx.Graph): graph to captureString diff lines
    """
    assert isinstance(graph, torch.fx.Graph)
    global graph_captured
    graph_captured = str(graph)


@disable_when(LOG_LEVEL > DEBUG)
def log(graph: torch.fx.Graph, title: str, recapture: bool):
    """
    Capture the end-point graph for graph-diff.
    String diff lines will be printed to debug logger if enabled.

    Args:
        graph (torch.fx.Graph): graph to capture
        title (str): Title in log
        recapture (bool): recapture the graph
    """
    assert isinstance(graph, torch.fx.Graph)
    global graph_captured

    logger = getLogger(__name__)
    diff = strdiff(f"{graph_captured}\n", f"{graph}\n")
    prefix = f"[{title}]" if title else ""
    if len(diff) > 0:
        logger.debug(f"{prefix} Graph is changed.")
        logger.debug(f"\n{diff}")

    if recapture:
        graph_captured = deepcopy(graph)
    else:
        graph_captured = None  # reset


# TODO diff graph signature
