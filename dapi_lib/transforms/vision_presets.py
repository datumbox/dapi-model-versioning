# These are example implementations similar to what we have on references.
# They will be adapted based on the work at https://github.com/pmeier/torchvision-datasets-rework/
import torch

from torch import Tensor, nn
from typing import Tuple
from torchvision import transforms as T
from torchvision.transforms import functional as F


# Allows handling of both PIL and Tensor images
class ConvertImageDtype(nn.Module):

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, img: Tensor) -> Tensor:
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        return F.convert_image_dtype(img, self.dtype)


class ImageNetEval(nn.Module):

    def __init__(self, crop_size: int, resize_size: int = 256, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225),
                 interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR) -> None:
        super().__init__()
        self.transforms = T.Compose([
            T.Resize(resize_size, interpolation=interpolation),
            T.CenterCrop(crop_size),
            ConvertImageDtype(dtype=torch.float),
            T.Normalize(mean=mean, std=std),
        ])

    def forward(self, img: Tensor) -> Tensor:
        return self.transforms(img)


class CocoEval(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.transforms = ConvertImageDtype(dtype=torch.float)

    def forward(self, img: Tensor) -> Tensor:
        return self.transforms(img)
