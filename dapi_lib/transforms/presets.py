# These are example implementations similar to what we have on references.
# Currently support only Tensor backend. Presets can be optionally nn.Modules and JIT-scriptable.
# They will be adapted based on the work at https://github.com/pmeier/torchvision-datasets-rework/
import torch

from torch import Tensor, nn
from typing import Tuple
from torchvision import transforms as T


class ImageNetEval(nn.Module):

    def __init__(self, crop_size: int, resize_size: int = 256, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR) -> None:
        super().__init__()
        self.transforms = T.Compose([
            T.Resize(resize_size, interpolation=interpolation),
            T.CenterCrop(crop_size),
            T.ConvertImageDtype(dtype=torch.float),
            T.Normalize(mean=mean, std=std),
        ])

    def forward(self, img: Tensor) -> Tensor:
        return self.transforms(img)


class CocoEval(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.transforms = T.ConvertImageDtype(dtype=torch.float)

    def forward(self, img: Tensor) -> Tensor:
        return self.transforms(img)
