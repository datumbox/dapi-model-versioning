import sys
import warnings

import contextlib
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision._internally_replaced_utils import load_state_dict_from_url


__all__ = ['ContextParams', 'Weights', 'get', 'list', 'register']


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


# Can be extended to support hierarchies
_MODEL_METHODS: Dict[str, Tuple[Callable, Optional[Weights]]] = {}


def get(name: str, weights: Optional[Weights] = None, use_latest: bool = True) -> Tuple[Callable, Optional[Weights]]:
    method, latest_weight = _MODEL_METHODS[name]
    if use_latest and weights is None:
        weights = latest_weight
    model = method(weights=weights)
    return model, weights


def list() -> List[str]:
    return sorted(_MODEL_METHODS.keys())


def register(fn):
    module = sys.modules[fn.__module__]
    if not hasattr(module, '__all__'):
        module.__all__ = []

    model_name = fn.__name__
    if model_name in _MODEL_METHODS:
        raise Exception(f"A model is already registered with key '{model_name}'.")
    module.__all__.append(model_name)

    sig = signature(fn)
    if 'weights' not in sig.parameters:
        raise Exception("The method is missing the mandatory 'weights' argument.")

    ann = signature(fn).parameters['weights'].annotation
    if isinstance(ann, type) and issubclass(ann, Weights):
        weights_class = ann
    else:
        # handle cases like Union[Optional, T]
        weights_class = None
        for t in get_args(ann):
            if isinstance(t, type) and issubclass(t, Weights):
                weights_class = t
                break

    latest_weight = None
    if weights_class is not None:
        module.__all__.append(weights_class.__name__)
        latest_weight = next(iter(weights_class.get_latest()), None)

    _MODEL_METHODS[model_name] = (fn, latest_weight)

    return fn
