"""
Scenario:
    1. Multi-weight and Multi-version support (BC)

Description:
    Provide different pre-trained weights for the same model version and allow for multiple code versions for the same
    model variant.

Example:
    https://github.com/pytorch/vision/issues/3995

    Providing different pre-trained weights for the same model version might be necessary because:
    1. We improved the previous weights by using a new training recipe.
    2. We provide additional weights trained on a different dataset (COCO vs Pascal VOC).

    Here we propose a mechanism which allows us to keep track of different weights with different meta-data and preset
    transforms using the same model builder method. We also propose a way to handle significant changes on the models
    by introducing new model builder methods.
"""

import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Optional

from dapi_lib.models._api import register, Weights, WeightEntry
from dapi_lib.transforms.vision_presets import ConvertImageDtype


__all__ = ['MySOTA']


# This module stores the main implementation of the architecture. Usually there is one such class per model,
# nevertheless it doesn't have to be the case.
class MySOTA(nn.Module):
    def __init__(self, num_classes: int = 1000, **kwargs: Any) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.flatten(start_dim=1))


# Each model weights class inherits from the Weight class and must provide all mandatory fields. These fields can be
# easily adapted to the needs of the project.
class MySOTAWeights(Weights):
    NOTHOTDOG = WeightEntry(
        'https://fake/models/not-hot-dog_weights.pth',  # Weight URL
        partial(ConvertImageDtype, dtype=torch.float16),  # Constructor for preprocessing transforms
        {'size': (32, 32), 'classes': ['not hotdog', 'hotdog']},  # Arbitrary Meta-Data associated with the weights
        True  # Flag that indicates whether it's the latest available weights for the specific Dataset/Taxonomy combo.
    )
    CATDOG_v1 = WeightEntry(
        'https://fake/models/catdog_weights_v1.pth',
        partial(ConvertImageDtype, dtype=torch.float32),
        {'size': (32, 32), 'classes': ['cat', 'dog']},
        False
    )
    CATDOG_v2 = WeightEntry(
        'https://fake/models/catdog_weights_v2.pth',
        partial(ConvertImageDtype, dtype=torch.float16),
        {'size': (64, 64), 'classes': ['cat', 'dog']},
        True
    )


# Each model variant (such as `resnet18`, `resnet50` etc) has its own public building method and receives an optional
# `weights` parameter. The type of the `weights` param is uniquely associated with the specific builder method. This
# makes it easier to document the available models and find all the available pre-trained weights via static analysis.
# Here we also show-case an optional registration mechanism that adds the builder and its weight class to the public
# API of the module.
@register
def mysota(weights: Optional[MySOTAWeights] = None, progress: bool = True, **kwargs: Any) -> MySOTA:
    # Confirm we got the right weights
    MySOTAWeights.check_type(weights)

    if weights is not None:
        kwargs['num_classes'] = len(weights.meta['classes'])

    model = MySOTA(**kwargs)

    if weights is not None and 'fake' not in weights.url:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


class MySOTAV2Weights(Weights):
    NOTHOTDOG = WeightEntry(
        'https://fake/models/not-hot-dog_weights_v2.pth',
        partial(ConvertImageDtype, dtype=torch.float16),
        {'size': (32, 32), 'classes': ['not hotdog', 'hotdog']},
        True
    )


# If significant changes are needed for a model, these should be added on a new model builder to maintain BC.
# Whether or not we will introduce a new `MySOTAV2` class depends on how similar the two are and it should be
# assessed on a case-by-case basis. See https://github.com/pytorch/vision/pull/1224 and
# https://github.com/pytorch/pytorch/blob/294db060/torch/nn/quantized/dynamic/modules/linear.py#L44-L49
@register
def mysota_v2(weights: Optional[MySOTAV2Weights] = None, progress: bool = True, **kwargs: Any) -> MySOTA:
    # Confirm we got the right weights
    MySOTAV2Weights.check_type(weights)

    if weights is not None:
        kwargs['num_classes'] = len(weights.meta['classes'])

    model = MySOTA(version=2, **kwargs)  # here we assume we keep the same class rather than creating a MySOTAV2

    if weights is not None and 'fake' not in weights.url:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


if __name__ == "__main__":
    m1 = mysota(MySOTAWeights.CATDOG_v1)
    v1 = sum(x.numel() for x in m1.parameters())

    m2 = mysota()
    v2 = sum(x.numel() for x in m1.parameters())
    assert v1 == v2

    mysota_v2(MySOTAV2Weights.NOTHOTDOG)
