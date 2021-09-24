import warnings

from functools import partial
from torch import nn
from typing import Any, Optional

from ._api import register, Weights
from ..datasets.mock import ImageNet
from ..transforms.vision_presets import ImageNetEval

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.transforms import InterpolationMode


__all__ = ['ResNet']


def _resnet_v1_builder(arch: str, weights: Optional[Weights], progress: bool, **kwargs: Any) -> nn.Module:
    # Configuration based on model variant
    if arch == 'resnet50':
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif arch == 'resnext101_32x8d':
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        block = Bottleneck
        layers = [3, 4, 23, 3]
    else:
        raise ValueError(f"Unsupported model type {arch}")

    # Adjust number of classes if necessary
    if weights is not None:
        kwargs['num_classes'] = len(weights.meta['classes'])

    # Initialize model
    model = ResNet(block, layers, **kwargs)

    # Optionally load weights
    if weights is not None:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


class ResNet50Weights(Weights):
    ImageNet1K_RefV1 = (
        'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        partial(ImageNetEval, crop_size=224),
        {'size': (224, 224), 'classes': ImageNet.classes,
         'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#resnext-50-32x4d'},
        False
    )
    ImageNet1K_RefV2 = (
        'https://download.pytorch.org/models/resnet50-0676ba61.pth',  # pretend these weights are different
        partial(ImageNetEval, crop_size=224, interpolation=InterpolationMode.BICUBIC),
        {'size': (224, 224), 'classes': ImageNet.classes, 'recipe': None},
        True
    )


@register
def resnet50(weights: Optional[ResNet50Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    # Backward compatibility for pretrained
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ResNet50Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    # Confirm we got the right weights
    ResNet50Weights.check_type(weights)

    return _resnet_v1_builder('resnet50', weights, progress, **kwargs)


class ResNext101Weights(Weights):
    ImageNet1K_RefV1 = (
        'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        partial(ImageNetEval, crop_size=224),
        {'size': (224, 224), 'classes': ImageNet.classes,
         'recipe': 'https://github.com/pytorch/vision/tree/main/references/classification#resnext-101-32x8d'},
        True
    )


@register
def resnext101_32x8d(weights: Optional[ResNext101Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    # Backward compatibility for pretrained
    if "pretrained" in kwargs:
        warnings.warn("The argument pretrained is deprecated, please use weights instead.")
        weights = ResNext101Weights.ImageNet1K_RefV1 if kwargs.pop("pretrained") else None

    # Confirm we got the right weights
    ResNext101Weights.check_type(weights)

    return _resnet_v1_builder('resnext101_32x8d', weights, progress, **kwargs)
