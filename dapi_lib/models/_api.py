import warnings

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision._internally_replaced_utils import load_state_dict_from_url


__all__ = ['Weights']


@dataclass
class Weights(Enum):
    url: str
    transforms: Callable
    meta: Dict[str, Any]
    latest: bool

    @classmethod
    def get_latest(cls) -> List:
        return [x for x in cls if x.latest]

    def state_dict(self, progress: bool) -> Dict[str, Any]:
        if not self.latest:
            warnings.warn(f"The selected weights are not the latest. For best performance "
                          f"choose one of the latest weights: {self.get_latest()}")
        return load_state_dict_from_url(self.url, progress=progress)

    def __repr__(self):
        return f"{self.__class__.__name__}.{self._name_}"
