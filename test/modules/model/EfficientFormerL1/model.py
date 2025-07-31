import timm
import torch
from test.modules.base import TestModuleBase


class EfficientFormerL1(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.model: timm.models.efficientformer.EfficientFormer = (
            timm.create_model("efficientformer_l1", pretrained=True).to("cpu").eval()
        )
        self.rtol = 1e-4
        self.atol = 1e-4

    def forward(self, x: torch.Tensor):
        return self.model.forward(x)

    def get_example_inputs(self):
        torch.manual_seed(1)
        return (torch.randn(1, 3, 224, 224),), {}
