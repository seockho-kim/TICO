from typing import Any, Dict, Tuple

import torch.nn as nn


class ExampleInputMixin:
    def get_example_inputs(self) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        raise NotImplementedError("Must implement get_example_inputs")


class DynamicShapesMixin:
    def get_dynamic_shapes(self) -> Dict[str, Tuple[int, ...]]:  # type: ignore[empty-body]
        pass


class TestModuleBase(nn.Module, ExampleInputMixin, DynamicShapesMixin):
    pass
