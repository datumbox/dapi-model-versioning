"""
Scenario:
    3. Code change which affects the model behaviour but architecture remains the same (BC-breaking)

Description:
    The model code has been updated in a BC-breaking way (training or inference) but the architecture remains the same.
    Existing pre-trained weights work fine with the new code.

Example:
    https://github.com/pytorch/vision/issues/2326
    https://github.com/pytorch/vision/pull/2954

    The original `FeaturePyramidNetwork` had a bug on its initialization and instead of initializing all of the modules
    of the model, it was doing it for only the direct children. Fixing the bug didn't affect the model architecture, in
    other words the old pre-trained weights continued to work fine on the new code. Nevertheless training a new model
    using the updated code leads to different results.

    Obviously when it comes to bug fixing, maintaining BC makes no sense. But let's assume we do want to introduce a
    BC-breaking change (perhaps a significant improvement on training or inference) but also allow users to access the
    old behaviour. One way to achieve this would be by introducing a new parameter on the constructor that affects the
    behaviour of the Class.

    Here we present an alternative approach that allows us to roll back to the previous behaviour by using Context
    Managers without introducing additional parameters to the Class.
"""
from torch import nn, Tensor
from typing import Optional

from dapi_lib.models._api import ContextParams, Weights


__all__ = ['BCBreaking', 'BCBreakingWeights', 'bc_model']


class BCBreaking(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        attr = ContextParams.get(self, 'init_attr', 'modules')
        for m in getattr(self, attr)():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class BCBreakingWeights(Weights):
    OLD = (
        'https://download.pytorch.org/models/old_weights.pth',
        None,
        {},
        False
    )
    NEW = (
        'https://fake/models/new_weights.pth',
        None,
        {},
        True
    )


def bc_model(weights: Optional[BCBreakingWeights] = None) -> nn.Module:
    with ContextParams(BCBreaking, weights is None, init_attr='children'):
        model = BCBreaking()

    if weights is not None and 'fake' not in weights.url:
        model.load_state_dict(weights.state_dict(progress=False))

    return model


if __name__ == "__main__":
    m = bc_model(BCBreakingWeights.NEW)
    assert sum(x.sum() for x in m.parameters()) == 0.0

    m = bc_model()
    assert sum(x.sum() for x in m.parameters()) != 0.0
