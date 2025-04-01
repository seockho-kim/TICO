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

from typing import Any, Callable, Dict, List

import numpy as np
import torch


def compute_peir(base: torch.Tensor, target: torch.Tensor):
    """
    Calculate the Peak Error to Interval Ratio (PEIR) between two tensors.

    This function computes the PEIR between two tensors using the formula:
        PEIR = max(abs(tensor1 - tensor2)) / (max(tensor1) - min(tensor2))
    """
    assert base.shape == target.shape, f"shape mismatch: {base.shape} != {target.shape}"
    base_tensor = base.numpy()
    target_tensor = target.numpy()
    assert (
        base_tensor.dtype == np.float32 and target_tensor.dtype == np.float32
    ), f"dtype should be float32: base({base_tensor.dtype}), target({target_tensor.dtype})"

    base_tensor = base_tensor.reshape(-1)
    target_tensor = target_tensor.reshape(-1)

    assert (
        base_tensor.shape == target_tensor.shape
    ), f"Shape mismatch: {base_tensor.shape} != {target_tensor.shape}"

    peak_error = np.max(np.absolute(target_tensor - base_tensor))
    interval = np.max(base_tensor) - np.min(base_tensor)
    peir = peak_error / interval  # pylint: disable=invalid-name

    min_value = min([base_tensor.min(), target_tensor.min()])
    max_value = max([base_tensor.max(), target_tensor.max()])

    interval = max_value - min_value
    interval = 1.0 if interval == 0.0 else interval  # Avoid zero interval

    return peir


class MetricCalculator:
    """
    Compute metrics including both built-in and custom metrics.

    metrics
        A list of metric names for comparison.
    custom_metrics
        A dictionary of metric names and corresponding callable functions for comparison.
        Example: {'mse': mean_squared_error, 'cosine_similarity': cosine_similarity_fn}
    """

    builtin_metrics = {
        "peir": compute_peir,
    }

    def __init__(
        self,
        metrics: List[str] = list(),
        custom_metrics: Dict[str, Callable] = dict(),
    ):
        self.metrics: Dict[str, Callable] = dict()

        for m in metrics:
            if m in self.builtin_metrics:
                self.metrics[m] = self.builtin_metrics[m]
            else:
                raise RuntimeError(f"Invalid metric: {m}")

        duplicates = set(self.metrics).intersection(custom_metrics.keys())
        if len(duplicates) != 0:
            raise RuntimeError(f"There are duplicate metrics: {duplicates}")

        self.metrics = self.metrics | custom_metrics

    def compute(
        self, output1: List[torch.Tensor], output2: List[torch.Tensor]
    ) -> Dict[str, List[Any]]:
        """
        Compute both built-in metrics (if provided) and custom metrics.

        Returns
        --------
        Dict[str, Any]
            A dictionary with metric names and their computed values.
        """
        results: Dict[str, List[Any]] = dict()

        # Compute built-in metrics
        if self.metrics is not None:
            for m in self.metrics:
                results[m] = list()
                for out1, out2 in zip(output1, output2):
                    results[m].append(self.builtin_metrics[m](out1, out2))

        return results
