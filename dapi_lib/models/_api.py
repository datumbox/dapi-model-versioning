import contextlib
import sys
import warnings

from dataclasses import dataclass, fields
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args

# Import a few stuff that we plan to keep as-is to avoid copy-pasting
from torchvision._internally_replaced_utils import load_state_dict_from_url


__all__ = ['Weights', 'WeightEntry', 'ContextParams', 'get', 'list', 'register']


@dataclass
class WeightEntry:
    """
    This class is used to group important attributes associated with the pre-trained weights.

    The current implementation is an illustration of how one can define the WeightEntry. Adding, removing and adapting
    the attributes to meet the needs of each library is essential. This example implementation suggests using the
    following attributes:
        url (str): The location where we find the weights. Can be adapted to facilitate integration with manifold.
        transforms (Callable): A callable that constructs the preprocessing method (or validation preset transforms)
            needed to use the model. The reason we attach a constructor method rather than an already constructed
            object is because the specific object might have memory (for example a tokenizer) and thus we want to delay
            initialization until needed.
        meta (Dict[str, Any]): Stores meta-data related to the weights of the model and its configuration. These can be
            informative attributes (for example the number of parameters/flops, recipe link/methods used in training
            etc), configuration parameters (for example the `num_classes` or `n_symbol`) needed to construct the model
            or important meta-data (for example the `classes` of a classification model) needed to use the model.
        latest (bool): An boolean indicator which encodes whether the specific set of weights is the best available for
            the given model/dataset/taxonomy combination. If `False`, the API shows a warning to the user prompting
            them to switch their weights to the latest ones.
    """
    url: str
    transforms: Callable
    meta: Dict[str, Any]
    latest: bool


class Weights(Enum):
    """
    This class is the parent class of all model weights. Each model building method receives an optional `weights`
    parameter with its associated pre-trained weights.

    The class inherits from `Enum` and its values should be of type `WeightEntry`.

    The use of Enums rather than strings to encode the weight information is a fundamental property of the API. Enums
    allow for better typing and IDE integration, work well with static analysis tools and make documenting the available
    options easier than strings. Finally keeping the attributes associated with the weights in code allows us to
    programmatically manipulate the meta-data of the pre-trained models, build automatically the docs and integrate
    easier with paperswithcode.com's model-index.
    """
    def __init__(self, value: WeightEntry):
        self._value_ = value

    @classmethod
    def check_type(cls, obj: Any) -> None:
        if obj is not None and not isinstance(obj, cls):
            raise TypeError(f"Invalid Weight class provided; expected {cls.__name__} "
                            f"but received {obj.__class__.__name__}.")

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

    def __getattr__(self, name):
        # Be able to fetch WeightEntry attributes directly f
        for f in fields(WeightEntry):
            if f.name == name:
                return object.__getattribute__(self.value, name)
        return super().__getattr__(name)


class ContextParams(contextlib.ContextDecorator):
    """
    This Context Manager allows us to temporarily overwrite the default constructor values of a Target Class or modify
    its behaviour without requiring adding extra arguments to its constructor. The target class uses the static `get()`
    method of this class to retrieve the hidden parameters and modify its behaviour.

    This class is not an essential part of the proposed Model-Versioning API. Nevertheless it is a useful utility which
    allows us to minimize the effects of BC-breaking changes to classes such as Layers and Modules. This context
    manager can be nested arbitrary many times and its effects are cumulative. The overwritten parameters have
    class-wide effects which remain in-effect for us long us we are within the context manager. See scenario 2 & 3 for
    how to use it.

    Args:
        target_class (type): the class for which we want to modify its behaviour.
        active (bool): a flag that indicates whether the context manager should be active. This is useful for turning
            on/off the context manager depending on a condition.
        **params: the named parameters and the values that we want to pass to the target class.
    """

    _QUEUE_NAME = "_queued_params"
    _OVERWRITE_NAME = "_overwritten_params"

    def __init__(self, target_class: type, active: bool, **params: Any):
        self.target_class = target_class
        self.active = active
        self.params = params

    @staticmethod
    def get(object: Any, key: str, default: Any) -> Any:
        params = getattr(object, ContextParams._OVERWRITE_NAME, None)
        return default if params is None else params.get(key, default)

    def __enter__(self):
        if self.active:
            queue = getattr(self.target_class, ContextParams._QUEUE_NAME, [])
            queue.append(self.params)

            overwrites = {}
            for p in queue:
                overwrites.update(p)

            setattr(self.target_class, ContextParams._QUEUE_NAME, queue)
            setattr(self.target_class, ContextParams._OVERWRITE_NAME, overwrites)

        return self

    def __exit__(self, *exc):
        if self.active:
            queue = getattr(self.target_class, ContextParams._QUEUE_NAME, None)
            if queue:
                if len(queue) > 1:
                    queue.pop()
                else:
                    delattr(self.target_class, ContextParams._QUEUE_NAME)

            if hasattr(self.target_class, ContextParams._OVERWRITE_NAME):
                delattr(self.target_class, ContextParams._OVERWRITE_NAME)

        return False


# Non-essential part of the proposed Model-Versioning API. Show-cases that the solution works with registration
# mechanisms similar to those introduced at the torchvision-datasets-rework repo. Can be extended to support
# hierarchies of models.
_MODEL_METHODS: Dict[str, Tuple[Callable, Optional[Weights]]] = {}


# Special type of internal enum that signals the use of the latest weights
class _LatestWeights(Weights):
    LATEST = WeightEntry(None, None, None, None)


def get(name: str, weights: Optional[Weights] = _LatestWeights.LATEST) -> Tuple[Callable, Optional[Weights]]:
    """
    Builds a model using the specified model builder name and weights. If no weights are specified, the implementation
    selects the latest available for this model. This is useful for users who want to access always the best available
    weights. Users who prefer stability should always specify the second parameter.

    Args:
        name (str): the name of the previously registered model builder method.
        weights (Optional[Weights]): the weights that we should use to initialize the model. If not defined then the
            first available Weight marked as latest will be selected. Passing `None` will cause no weights to be loaded.

    Returns:
        Tuple[Callable, Optional[Weights]]: The model along with the weights enum used to initialize it.
    """
    method, latest_weight = _MODEL_METHODS[name]
    if weights == _LatestWeights.LATEST:
        weights = latest_weight
    model = method(weights=weights)
    return model, weights


def list() -> List[str]:
    """
    Lists all the registered model building methods.

    Returns:
        List[str]: The list of registered model builders.
    """
    return sorted(_MODEL_METHODS.keys())


def register(fn):
    """
    Adds the provided model building method along with its weights class to the public API. The method registers
    not only the function but also its latest weight (the first one if multiple).

    Args:
        fn (function): the model builder method that we want to register to the model API. It is assumed to have
            a `weights` parameter where the user can optionally pass its weights.

    Returns:
        function: The registered function.
    """
    module = sys.modules[fn.__module__]
    if not hasattr(module, '__all__'):
        module.__all__ = []

    method_name = fn.__name__
    if method_name in _MODEL_METHODS:
        raise Exception(f"A method is already registered with key '{method_name}'.")
    module.__all__.append(method_name)

    sig = signature(fn)
    if 'weights' not in sig.parameters:
        raise Exception("The method is missing the 'weights' argument.")

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

    _MODEL_METHODS[method_name] = (fn, latest_weight)

    return fn
