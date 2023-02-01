import torch.nn as nn
from torch import Tensor
from mmyolo.models import RepVGGBlock
from mmdet.models.backbones.csp_darknet import Focus
from .common import TRTFocus, NCNNFocus, GConvFocus


class WarpperModel(nn.Module):
    ForceTranspose = True

    def __init__(self, baseModel: nn.Module, backend: str = 'onnxruntime'):
        super().__init__()
        self.baseModel = baseModel
        self.mean = baseModel.data_preprocessor.mean
        self.std = baseModel.data_preprocessor.std
        self.backend = backend
        self.__switch_deploy()

    def __switch_deploy(self):
        for layer in self.baseModel.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, Focus):
                if self.backend == 'tensorrt' or \
                        self.backend == 'onnxruntime':
                    self.baseModel.backbone.stem = TRTFocus(layer)
                elif self.backend == 'ncnn':
                    self.baseModel.backbone.stem = NCNNFocus(layer)
                else:
                    self.baseModel.backbone.stem = GConvFocus(layer)

    def forward(self, inputs: Tensor):
        neck_outputs = self.baseModel(inputs)
        if self.ForceTranspose:
            neck_outputs = [
                [b.permute(0, 2, 3, 1),
                 c.permute(0, 2, 3, 1)]
                for b, c in zip(*neck_outputs)]
        return sum(neck_outputs, [])
