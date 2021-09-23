import warnings

import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision._internally_replaced_utils import load_state_dict_from_url


__all__ = ['ContextParams', 'Weights']


class ContextParams(contextlib.ContextDecorator):

    _QUEUE_NAME = "_queued_params"
    _OVERWRITE_NAME = "_overwritten_params"

    def __init__(self, klass: type, active: bool, **kwargs: Any):
        self.klass = klass
        self.active = active
        self.params = kwargs

    @staticmethod
    def get(object: Any, key: str, default: Any) -> Any:
        params = getattr(object, ContextParams._OVERWRITE_NAME, None)
        return default if params is None else params.get(key, default)

    def __enter__(self):
        if self.active:
            queue = getattr(self.klass, ContextParams._QUEUE_NAME, [])
            queue.append(self.params)

            overwrites = {}
            for p in queue:
                overwrites.update(p)

            setattr(self.klass, ContextParams._QUEUE_NAME, queue)
            setattr(self.klass, ContextParams._OVERWRITE_NAME, overwrites)

        return self

    def __exit__(self, *exc):
        if self.active:
            queue = getattr(self.klass, ContextParams._QUEUE_NAME, None)
            if queue:
                if len(queue) > 1:
                    queue.pop()
                else:
                    delattr(self.klass, ContextParams._QUEUE_NAME)

            if hasattr(self.klass, ContextParams._OVERWRITE_NAME):
                delattr(self.klass, ContextParams._OVERWRITE_NAME)

        return False


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
