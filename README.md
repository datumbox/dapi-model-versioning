# DAPI Model Versioning RFC

1. [Introduction](#introduction)
    1. [Problem definition](#problem-definition)
    2. [Objective](#objective)
    3. [Motivation](#motivation)
2. [Previous work](#previous-work)
3. [Repository structure](#repository-structure)
4. [Design](#design)
    1. [Specifications](#specifications)
    2. [Out of scope](#out-of-scope)
    3. [Proposal](#proposal)
    3. [Special cases](#special-cases)
    4. [Demos](#demos)
    5. [Implementation details](#implementation-details)
    6. [Alternatives considered](#alternatives-considered)
    7. [Release plan](#release-plan)
5. [Next steps](#next-steps)

## Introduction

### Problem definition

The PyTorch domain libraries don't have a standard way to perform Model Versioning. With the term "Model Versioning" we 
denote the problem of maintaining concurrently multiple versions of pre-trained weights and handling changes on the 
model code both in a Backwards Compatible (BC) and BC-breaking manner.

### Objective

Establish a common approach for handling Model Versioning across all domain libraries. 

### Motivation

Currently all domain libraries offer comparable APIs for initializing models of popular architectures. Those that 
already offer pre-trained weights (TorchAudio and TorchVision) have adopted equivalent solutions 
[[1](https://pytorch.org/audio/main/models.html#torchaudio.models.tacotron2), 
[2](https://pytorch.org/vision/stable/models.html#id10)] while those that plan to introduce them for the first time 
(TorchText) aligned with existing practices [[3](https://www.internalfb.com/intern/anp/view/?id=1138838), 
[4](https://www.internalfb.com/tasks/?t=100593038)]. 

While the model building process is fairly standardized across domains, the model versioning isn't. Fortunately, the
needs across all domains are very similar. We all have to support multiple weights, handle model code changes, strive 
for reproducibility etc. Standardizing the way we do this across domains is important because:
1. Code changes affecting the models are common and part of the standard development cycle of all domains.
2. Updating the model weights with more accurate ones trained using new recipes is a common problem 
   [[5](https://github.com/pytorch/vision/issues/3995)]. Same applies to providing concurrently multiple versions of 
   the weights trained on different Datasets (to cover for different taxonomies, languages etc) 
   [[6](https://github.com/pytorch/audio/blob/352d63c5/torchaudio/models/tacotron2.py#L70-L97)].
3. Providing an aligned API will improve user experience while combining the Domain libraries in their training recipes.

## Previous work

This is not the first time the model versioning problem arises [[7](https://github.com/pytorch/vision/issues/2955)] 
as it has previously been discussed by the domain library maintainers. 

In TorchVision, model versioning related issues have been handled on a case-by-case basis 
[[8](https://github.com/pytorch/vision/issues/2599), [9](https://github.com/pytorch/vision/issues/2326), 
[10](https://github.com/pytorch/vision/pull/3205), [11](https://github.com/pytorch/vision/pull/1224), 
[12](https://github.com/pytorch/vision/pull/4487)]. We typically try to maintain BC as much as possible except in cases 
where the issue is considered a bug or extremely detrimental to the user experience. In the latter cases, we often 
deploy BC-breaking fixes but try to reduce the effects of the change as much as possible 
[[13](https://github.com/pytorch/vision/pull/2940)]. TorchAudio has only recently introduced the pre-trained models and 
their API is currently in beta. TorchText is currently working on introducing its first pre-trained models so their 
solution is in prototype phase. Finally, though PyTorch Core doesn't provide pre-trained models, occasionally there is 
the need of handling code changes on existing Layers and operators. These are typically handled with a mix of version 
parameters, deprecation warnings and method renamings
[[14](https://github.com/pytorch/pytorch/blob/294db060/torch/nn/quantized/dynamic/modules/linear.py#L44-L49),
[15](https://github.com/pytorch/pytorch/blob/c371542e/caffe2/python/cnn.py#L182-L183),
[16](https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/caffe2/python/brew.py#L65-L66)]

## Repository structure

This RFC comes with a companion [repository](https://github.com/datumbox/dapi-model-versioning). Amendments to this 
RFC should be made by sending a Pull Request to the repo.

The repository aims to serve as a live RFC document capable of show-casing the proposed API and utilities, providing 
examples of how to address the most common model-versioning scenarios and offering actual implementations for some of 
the real-world models included in the Domain libraries. Here is its structure:

- The `README.md` file serves as the main RFC document.
- The `examples` folder contains standalone implementations for the most common model versioning scenarios that we've 
  faced before. On the top of each scenario file we include a description and an example with references to a real-world
  case. We recommend starting from there.
- The actual implementation of the proposal lives in the `dapi_lib` package:
    - The `dapi_lib/models/_api.py` contains the majority of the utilities used by the API.
    - The rest of the files in `dapi_lib/models/*.py` are implementations of real-world models from different domains.
    - The `dapi_lib/datasets` and `dapi_lib/transforms` packages contain code taken from the domain libs and adapted
    for the needs of this RFC. These are purely there to make the demos run smoothly and they should not be considered
    part of this proposal.
    - Extensive importing from the Domain libraries and inheritance is used to minimize the copy-pasting of code from
    the domain libs. The intention is to upstream the proposed changes after the RFC concludes.
- The real-world demos are placed in the `*.py` files located at the root of the repo. They target to show how the API 
  looks from the user perspective:
    - The `image_classification.py` and `image_detection.py` show-case the new API on Vision.
    - The `text_encoding.py` gives us an example on how Text could structure its models. Note that because TorchText
    currently doesn't provide pre-trained models on the public repo, we use huggingface's Roberta.
    - The `text_to_speech.py` provides an example of implementing the new API on Audio.
- Other details:
    - The `assets` folder contains a couple of assets necessary for the demos.
    - An `output` folder will be created on the root of the project after running the demos.
    - The `third_party` package contains copy-pasted code necessary for the demos and it is not part of this RFC.
    - The `requirements.txt` file contains a list of all dependencies for running the code on this repo.

## Design

### Specifications

The proposed solution must meet the following criteria:
1. Provide an API for supporting multiple pre-trained weights and the paradigm of how to version and handle the code 
   changes on models.
2. Describe how the pre-trained weights link to their corresponding meta-data and to the preprocessing transforms 
   which are necessary for using the models.
3. Facilitate the discoverability of the various model variants (example: `resnet50`) along with the available 
   pre-trained weights by reusing as many standard dev tools from the python ecosystem as possible.

The proposed solution can optionally support the following nice-to-haves:
1. Encourage users to use the latest/greatest pre-trained weights and models.
2. Provide utilities that simplify model-versioning in BC cases and reduce side-effects in BC-breaking cases.

### Out of scope

1. We focus on the main model building methods used for initializing the models and loading their weights. Libraries 
   can also offer secondary options for doing so (such as constructing models directly from Classes, supporting model 
   registration mechanisms etc) but these are out of scope for this proposal. Note that when possible, we illustrate
   with code examples that our solution is compatible with such mechanisms, but they are not part of our proposal.
2. We focus on the public API and not on implementation details. Though the repo contains private helper methods used to
   [construct models](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/resnet.py#L19-L43),
   these are not part of the proposal and libraries can adapt them to their needs and current practices.
3. We try to keep this RFC not too opinionated but rather describe a framework that gives to the DAPI libs the space to
   adapt the solution on their needs. As a result, fully specifying implementation details is beyond the scope of this
   proposal.

### Proposal

We propose using separate model builders and a weights parameter for each model version. We plan to maintain the 
existing model builder methods supported by all Domain libraries to construct models and use Enums with data class 
values to pass the information of the pre-trained weights. Each model variant will have its own method and weights. When 
BC-breaking changes are necessary, we will introduce new builder methods to keep things BC. 

High-level API implementation in pseudocode:

```python
@dataclass
class WeightEntry:
    url: str  # Weights URL/path
    transforms: Callable  # Preprocessing transform constructor
    meta: Dict[str, Any]  # Arbitrary Meta-Data
    # Other customizable fields go here

class ResNet50Weights(Enum):
    ImageNet1K_RefV1 = WeightEntry(
        url="https://path/to/weights.pth",
        transforms=partial(ImageNetPreprocessing, width=224),
        meta={"num_classes": 1000, "Acc@1": 76.130, "classes": [...]}
    )
    CIFAR100_RefV1 = WeightEntry(...)

def resnet50(weights: Optional[ResNet50Weights] = None) -> ResNet:
    # Model construction and load weighting goes here
    pass

# When BC-breaking changes are unavoidable, we will provide new builder methods to keep things BC.
class ResNet50V2Weights(Enum):
    ImageNet1K_RefV1 = WeightEntry(...)

def resnet50_v2(weights: Optional[ResNet50V2Weights] = None) -> ResNetV2:  # Assume new Class needed
    pass
```

Example of using the API:
```python
# Model initialization and weights loading
weights = ResNet50Weights.ImageNet1K_RefV1
model = resnet50(weights=weights).eval()

# Fetch preprocessing transforms and apply them to the image
preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0)

# Make predictions
prediction = model(batch).squeeze(0).softmax(0)

# Use meta-data
label = prediction.argmax().item()
class_name = weights.meta['classes'][label]
```

The above approach:
- Can be implemented in a fully BC manner and it's easy to adopt across all domain libraries.
- Covers all mandatory criteria.
- Supports all the nice-to-haves.

### Special cases

This section provides some guidance on how to handle special cases that were brought up during the review of the RFC. 
When multiple valid options exist, the DAPI libs should choose the one that meets their needs best:
- If a model requires both pre-processing and post-processing transforms, there are two main options (for more details 
  see [issue #5](https://github.com/datumbox/dapi-model-versioning/issues/5)):
    1. We can add both of them in a single transformer class, which will expose 2 public methods one for `preprocessing` 
       and one for `postprocessing`.
    2. We can provide two separate transformer classes which will implement `preprocessing` and `postprocessing` on 
       their `forward()`/`__call()__` method. Then we will offer two separate fields for them on the `WeightEntry` 
       class.
- Models often use other models as backbones (for more details 
  see [issue #6](https://github.com/datumbox/dapi-model-versioning/issues/6)). This paradigm is common on Audio, Text
  and Vision and usually encoder architectures are combined with extra Heads to solve new tasks. In this RFC, we propose 
  to use model builders to define combinations of encoders and extra heads and we provide a demo of 
  [how to implement it](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/faster_rcnn.py#L77-L80). 
  Supporting arbitrary combinations of encoders and task heads is also possible by passing the `backbone` to the 
  [composite Model classes](https://github.com/datumbox/dapi-model-versioning/issues/6#issuecomment-932251484) but this
  topic is beyond the scope of this RFC.

### Demos

To prove that the proposed API can accommodate all domains, we implemented it to 4 real-world models. To see the demos
run the following commands from the root of the repo:

#### Image Classification: ResNet50

Check the [model implementation](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/resnet.py) 
and the [model usage](https://github.com/datumbox/dapi-model-versioning/blob/main/image_classification.py).

```
$ python -u image_classification.py
golden retriever 0.9381255507469177
```

#### Object Detection: FasterRCNN with ResNet50 FPN

Check the [model implementation](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/faster_rcnn.py) 
and the [model usage](https://github.com/datumbox/dapi-model-versioning/blob/main/image_detection.py).

```
$ python -u image_detection.py.py
Saving picture at ./output/object-detection.jpg
```
![object detection](https://i.ibb.co/yPQXxmf/object-detection.jpg)

#### Text Encoding: Roberta Base

Check the [model implementation](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/roberta.py) 
and the [model usage](https://github.com/datumbox/dapi-model-versioning/blob/main/text_encoding.py).

```
$ python -u text_encoding.py
125000000 torch.Size([1, 5, 768])
```

#### Text to Speech: Tacotron2 + WaveRNN

Check the [model implementation](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/tacotron2.py) 
and the [model usage](https://github.com/datumbox/dapi-model-versioning/blob/main/text_to_speech.py).

```
$ python -u text_to_speech.py 
38 torch.Size([1, 80, 112])
Saving wave at ./output/message.wav
```

### Implementation details

Below we link directly to the actual implementations and code examples where we document everything extensively. The 
best way to see how the proposal works is to check the `examples` folder, where we focus on 3 Model Versioning scenarios
that we had to address in the past:
1. [Multi-weight and Multi-version support (BC)](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario1.py)
2. [Updated default Hyper-params (BC-breaking)](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario2.py)
3. [Code change which affects the model behaviour but architecture remains the same (BC-breaking)](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario3.py)

Our proposal consists of the following key components:
- The [Weights](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L47-L87) Enum and
  the [WeightEntry](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L17-L44) data
  class which store crucial information about the pre-trained weights.
- The [model builder methods](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario1.py#L78-L96)
  which construct the model variants and load the pre-trained weights.

We also offer two optional components:
- The [ContextParams](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L90-L148) 
  class which allows us to minimize the effects of BC-breaking changes to classes such as Layers and Modules.
- A [Registration](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L151-L236)
  mechanism similar to the one used on the prototype datasets of 
  [TorchVision](https://github.com/pytorch/vision/blob/main/torchvision/prototype/datasets/_api.py) to show-case that
  our proposal is compatible with it.

### Alternatives considered

Here we briefly list the alternatives that we considered along with some of the reasons we didn't select them. Note that
in all cases, we prefer using Enums to strings. To read more on why check this 
[section](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L54-L56).

#### Single model builder and weights parameter for all code versions

```python
class ResNet50Weights(Enum):
    V1_NoWeights = WeightEntry(...)
    V1_ImageNet1K_RefV1 = WeightEntry(...)

def resnet50(weights: ResNet50Weights = ResNet50Weights.V1_NoWeights) -> nn.Module:
    pass
```

Pros:
- Small number of model builders and Enum classes to maintain.

Cons:
- All versions must be handled in a single method leading to complex implementations.
- Harder to document and unit-test using standard python tools.
- Since the version is linked to the weights enum, it would require the introduction of special enums to denote that no 
  pre-trained weights should be loaded.

#### Single model builder, two separate arguments for the version and weights

```python
class ResNet50Weights(Enum):
    ImageNet1K_RefV1 = WeightEntry(...)

def resnet50(version: int = 1, weights: Optional[ResNet50Weights] = None) -> nn.Module:
    pass
```

Pros:
- Small number of model builders and Enum classes to maintain.

Cons:
- All versions must be handled in a single method leading to complex implementations.
- Harder to document and unit-test using standard python tools.
- Difficult for the users to tell which `version` is compatible with which `weights` enum.

#### Separate model builder for each code version and weights combination

```python
def resnet50_v2_imagenet_ref1(pretrained: bool = False) -> ResNetV2:
    pass
```

Pros:
- No changes required. Keeps things as they are in domain libraries.
- Keeps things immutable and frozen in time.

Cons:
- The number of methods increases multiplicatively with each version, dataset and recipe combination.
- Can lead to a lot of legacy code.
- It's a nonsolution. It does not really addresses the Versioning problem.

### Release plan

The key components of the proposal should be independently adapted and implemented by the maintainers of each domain 
library. The degree to which the optional components will be adopted can vary and depends on the needs of each domain.
Since we currently don't have an agreed way to handle interdependencies between domain libraries, any of the proposed 
utilities adopted can live in multiple domain repos and can move on a common repo if such is introduced on the future.

## Next steps

- [ ] Request input from TorchAudio and TorchText leads and adapt the proposal based on their feedback. 
- [ ] Present the proposal to the broad DAPI maintainers and make amendments based on their input. 
- [ ] Collect feedback from the community and iterate on the proposal.
- [ ] Implement the proposal on domain libraries.
