# Copyright (c) 2024 Intel Corporation
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

import types
from typing import Any, Dict, List, Optional

import torch

from tico.experimental.quantization.algorithm.gptq.gptq import GPTQ
from tico.experimental.quantization.algorithm.gptq.utils import (
    find_layers,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
)
from tico.experimental.quantization.config import BaseConfig, GPTQConfig
from tico.experimental.quantization.quantizer import BaseQuantizer


class GPTQQuantizer(BaseQuantizer):
    """
    Quantizer for applying the GPTQ algorithm (typically for weight quantization)
    """

    def __init__(self, config: BaseConfig):
        super().__init__(config)

        self.cache_args: List[Any] = []
        self.cache_kwargs: Dict[str, Any] = {"batch_num": 0}

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Overrides the forward method of the first LLaMA layer (layer 0) to capture the
         input required for calibration.

        This method modifies the original forward pass of LLaMA layer 0 so that the
         inputs used during inference are intercepted and recorded. These captured inputs
        are then utilized to calibrate the quantization parameters for the GPTQ.

        Parameters:
            model: The target PyTorch model.
            args: Positional example inputs required for capturing graph.
            kwargs: Keyword example inputs required for capturing graph.

        Returns:
            The model prepared for GPTQ quantization.
        """
        if args is None and kwargs is None:
            raise RuntimeError(
                "Either args or kwargs must be provided for captruing graph."
            )
        # Define a function to capture input activations and associated parameters.
        def forward(layer, *args, **kwargs):
            self.cache_kwargs["batch_num"] += 1
            for idx, item in enumerate(args):
                if (idx + 1) > len(self.cache_args):
                    self.cache_args.append([])
                self.cache_args[idx].append(item)
            for arg in kwargs:
                if self.cache_kwargs.get(arg, None) is None:
                    self.cache_kwargs[arg] = []
                self.cache_kwargs[arg].append(kwargs[arg])
            # Raise an error to interrupt the forward pass after capturing data.
            raise ValueError

        # Replace the first layer with defined function to capture calibration data.
        if hasattr(model, "model"):
            assert hasattr(model.model, "layers")
            assert isinstance(model.model.layers, torch.nn.ModuleList)
            layer_forward_cache = model.model.layers[0].forward
            model.model.layers[0].forward = types.MethodType(
                forward, model.model.layers[0]
            )
        else:
            assert hasattr(model, "forward")
            layer_forward_cache = model.forward
            model.forward = types.MethodType(forward, model.forward)

        model_forward_cache = model.forward
        # Replace model's forward to avoid ValueError
        def model_forward(model, *args, **kwargs):
            nonlocal model_forward_cache
            try:
                model_forward_cache(*args, **kwargs)
            except ValueError:
                pass

        model.forward = types.MethodType(model_forward, model)
        kwargs = kwargs or {}
        model(*args, **kwargs)  # type: ignore[misc]

        # Recover original forward
        model.forward = model_forward_cache
        if hasattr(model, "model"):
            assert hasattr(model.model, "layers")
            assert isinstance(model.model.layers, torch.nn.ModuleList)
            model.model.layers[0].forward = layer_forward_cache
        else:
            model.forward = layer_forward_cache

        return model

    @torch.no_grad()
    def convert(self, model):
        """
        Convert the prepared model to its GPTQ quantized version.

        Applies the GPTQ quantization on weights based on the collected statistics.

        Parameters:
            model: The prepared PyTorch model.

        Returns:
            The quantized model.
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, GPTQConfig)

        # Save the original cache setting and disable caching during calibration/inference.
        if hasattr(model, "config"):
            use_cache = model.config.use_cache
            model.config.use_cache = False

        quantizers = {}
        if hasattr(model, "model"):
            target_layers = model.model.layers
        else:
            target_layers = [model]
        for l_idx, layer in enumerate(target_layers):
            # Identify quantizable submodules within the layer.
            full = find_layers(layer)

            sequential = [list(full.keys())]
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq: Dict[str, GPTQ] = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        8, perchannel=True, sym=False, mse=False
                    )
                # Define a hook to collect input/output batches for quantizer calibration.
                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                # Run the current layer on the stored calibration inputs to capture activation stats.
                batch_num = self.cache_kwargs.pop("batch_num")
                for batch_idx in range(batch_num):
                    cache_args_batch = gather_single_batch_from_list(
                        self.cache_args, batch_idx
                    )
                    cache_kwargs_batch = gather_single_batch_from_dict(
                        self.cache_kwargs, batch_idx
                    )
                    layer(*cache_args_batch, **cache_kwargs_batch)[0]
                self.cache_kwargs["batch_num"] = batch_num
                for h in handles:
                    h.remove()
                # Quantize each submodule using the collected calibration data.
                for name in subset:
                    if gptq_conf.verbose:
                        print(l_idx, name)
                        print("Quantizing ...")
                    gptq[name].fasterquant(
                        percdamp=0.01,
                        groupsize=-1,
                        actorder=True,
                        static_groups=False,
                        verbose=gptq_conf.verbose,
                    )
                    quantizers["model.layers.%d.%s" % (l_idx, name)] = gptq[
                        name
                    ].quantizer
                    gptq[name].free()
            """
            Execute the quantized layer with the calibration inputs to obtain ouptuts
             that will serve as inputs for the next layer.

            This ensures that the quantization effects are correctly propagated to subsequent
             layers.
            """
            batch_num = self.cache_kwargs.pop("batch_num")
            for batch_idx in range(batch_num):
                cache_args_batch = gather_single_batch_from_list(
                    self.cache_args, batch_idx
                )
                cache_kwargs_batch = gather_single_batch_from_dict(
                    self.cache_kwargs, batch_idx
                )
                outs = layer(*cache_args_batch, **cache_kwargs_batch)[0]
                # Update inputs for next iteration.
                self.cache_args[0][batch_idx] = outs
            self.cache_kwargs["batch_num"] = batch_num

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Restore the original cache configuration.
        if hasattr(model, "config"):
            model.config.use_cache = use_cache

        return model
