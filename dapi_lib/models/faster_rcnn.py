import warnings

from typing import Any, Optional

from . import resnet
from ._api import register, ContextParams, Weights, WeightEntry
from .resnet import ResNet50Weights
from ..datasets.mock import Coco
from ..transforms.vision_presets import CocoEval

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN, _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


__all__ = ['FasterRCNN']


# Inherit to avoid copy-pasting the whole class. The changes should be upstreamed to the parent class.
class FrozenBatchNorm2d(misc_nn_ops.FrozenBatchNorm2d):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(num_features, eps=ContextParams.get(self, 'eps', eps))



def _resnet_fpn_backbone(
    backbone_name,
    weights,
    norm_layer=FrozenBatchNorm2d,
    trainable_layers=3,
    returned_layers=None,
    extra_blocks=None
):
    backbone = resnet.__dict__[backbone_name](
        weights=weights,
        norm_layer=norm_layer)

    # COPY-PASTED CODE FROM torchvision.models.detection.backbone_utils.resnet_fpn_backbone
    # =====================================================================================
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append('bn1')
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    # =====================================================================================


class FasterRCNNResNet50FPNWeights(Weights):
    Coco_RefV1 = WeightEntry(
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
        CocoEval,
        {'classes': Coco.classes,
         'recipe': 'https://github.com/pytorch/vision/tree/main/references/detection#faster-r-cnn-resnet-50-fpn'},
        True
    )


@register
def fasterrcnn_resnet50_fpn(weights: Optional[FasterRCNNResNet50FPNWeights] = None,
                            weights_backbone: Optional[ResNet50Weights] = ResNet50Weights.ImageNet1K_RefV1,
                            progress: bool = True, num_classes: int = 91,
                            trainable_backbone_layers: Optional[int] = None, **kwargs: Any) -> FasterRCNN:
    # Backward compatibility for pretrained
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = FasterRCNNResNet50FPNWeights.Coco_RefV1 if kwargs.pop("pretrained") else None
    if "pretrained_backbone" in kwargs:
        warnings.warn("The argument pretrained_backbone is deprecated, please use weights_backbone instead.")
        weights_backbone = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained_backbone") else None

    # Confirm we got the right weights
    FasterRCNNResNet50FPNWeights.check_type(weights)
    ResNet50Weights.check_type(weights_backbone)

    if weights is not None:
        # No need to download the backbone weights
        weights_backbone = None

        # Adjust number of classes if necessary
        num_classes = len(weights.meta['classes'])

    trainable_backbone_layers = _validate_trainable_layers(
        weights is not None or weights_backbone is not None, trainable_backbone_layers, 5, 3)

    # Overwrite the default eps value. See scenario 2 for full explanation.
    with ContextParams(FrozenBatchNorm2d, weights is not None, eps=0.0):
        backbone = _resnet_fpn_backbone('resnet50', weights_backbone, trainable_layers=trainable_backbone_layers)
        model = FasterRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model
