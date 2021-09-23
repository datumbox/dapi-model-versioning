"""
Scenario:
    1. Same code, different model weights (BC)

Description:
    Provide different pre-trained weights for the same model. All default hyper-params and code remain the same.

Example:
    https://github.com/pytorch/vision/issues/3995

    Providing different pre-trained weights might be necessary because:
    1. We improved the previous weights by using a new training recipe.
    2. We provide additional weights trained on a different dataset (COCO vs Pascal VOC).

    Here we propose a mechanism which allows us to keep track of different weights with different meta-data and preset
    transforms using the same model builder method.
"""

import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Optional

from dapi_lib.models._api import Weights
from dapi_lib.transforms.vision_presets import ConvertImageDtype


__all__ = ['MySOTA', 'MySOTAWeights', 'mysota']


class MySOTA(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
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


class MySOTAWeights(Weights):
    NOTHOTDOG = (
        'https://fake/models/not-hot-dog_weights.pth',
        partial(ConvertImageDtype, dtype=torch.float16),
        {'size': (32, 32), 'classes': ['not hotdog', 'hotdog']},
        True
    )
    CATDOG_v1 = (
        'https://fake/models/catdog_weights_v1.pth',
        partial(ConvertImageDtype, dtype=torch.float32),
        {'size': (32, 32), 'classes': ['cat', 'dog']},
        False
    )
    CATDOG_v2 = (
        'https://fake/models/catdog_weights_v2.pth',
        partial(ConvertImageDtype, dtype=torch.float16),
        {'size': (64, 64), 'classes': ['cat', 'dog']},
        True
    )


def mysota(weights: Optional[MySOTAWeights] = None, progress: bool = True, **kwargs: Any) -> nn.Module:
    if weights is not None:
        kwargs['num_classes'] = len(weights.meta['classes'])

    model = MySOTA(**kwargs)

    if weights is not None and 'fake' not in weights.url:
        model.load_state_dict(weights.state_dict(progress=progress))

    return model


if __name__ == "__main__":
    m1 = mysota(MySOTAWeights.CATDOG_v1)
    v1 = sum(x.numel() for x in m1.parameters())

    m2 = mysota()
    v2 = sum(x.numel() for x in m1.parameters())
    assert v1 == v2
