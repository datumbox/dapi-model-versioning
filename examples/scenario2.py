"""
Scenario:
    2. Updated default Hyper-params (BC-breaking)

Description:
    The default hyper-param of a class/method/layer needs to be changed (BC-breaking) because it causes issues to the
    users. All other code remains the same. New models need to be constructed with the updated value, old pre-trained
    models must continue using the original value.

Example:
    https://github.com/pytorch/vision/issues/2599
    https://github.com/pytorch/vision/pull/2933
    https://github.com/pytorch/vision/pull/2940

    The original value of eps of the `FrozenBatchNorm2d` was `0.0` but was causing training stability problems.
    We considered it a bug and thus we BC-broke by updating the default value to `1e-5` in the class. Nevertheless
    previously trained models had to continue using `0.0`. To resolve it we introduced the method
    `torchvision.models.detection._utils.overwrite_eps()` to overwrite the epsilon values of all FrozenBN layers
    after they have been created.

    Here we propose an alternative mechanism which allows to overwrite the default values during object construction
    using Context Managers.
"""
from torch import nn, Tensor
from typing import Optional

from dapi_lib.models._api import register, ContextParams, Weights

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision.ops.misc import FrozenBatchNorm2d


__all__ = ['Dummy']


# Note: The only reason why we inherit instead of making the changes directly to FrozenBatchNorm2d is to avoid
# copy pasting a lot of code from TorchVision. The changes below should happen on the parent class.
class MyFrozenBN(FrozenBatchNorm2d):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__(num_features, eps=ContextParams.get(self, 'eps', eps))


class Dummy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(32, 64, 1)
        self.bn = MyFrozenBN(64)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class DummyWeights(Weights):
    DUMMY = (
        'https://fake/models/dummy_weights.pth',
        None,
        {},
        True
    )


@register
def dummy(weights: Optional[DummyWeights] = None) -> nn.Module:
    with ContextParams(MyFrozenBN, weights is not None, eps=0.0):
        model = Dummy()

    if weights is not None and 'fake' not in weights.url:
        model.load_state_dict(weights.state_dict(progress=False))

    return model


if __name__ == "__main__":
    m = dummy(weights=DummyWeights.DUMMY)
    assert m.bn.eps == 0.0

    m = dummy()
    assert m.bn.eps == 1e-5
