# DAPI Model Versioning

## Introduction

### Problem definition

The PyTorch domain libraries don't have a standard way to perform Model Versioning. With the term "Model Versioning" we 
denote the problem of maintaining concurrently multiple versions of pre-trained weights and handling changes on the 
model code both in a Bacwards Compatible (BC) and BC-breaking manner.

### Objective

Establish a common approach for hanling Model Versioning across all domain libraries. 

### Motivation

Currently all domain libraries offer comparable APIs for initializing models of popular architectures. Those that 
already offer pre-trained weights (TorchAudio and TorchVision) have adopted equivalent solutions 
[ [1](https://pytorch.org/audio/main/models.html#torchaudio.models.tacotron2), 
[2](https://pytorch.org/vision/stable/models.html#id10) ] while those that plan to introduce them for the first time 
(TorchText) aligned with existing practices [ [3](https://www.internalfb.com/intern/anp/view/?id=1138838), 
[4](https://www.internalfb.com/tasks/?t=100593038) ]. 

While the model building process is fairly standardized across domains, the model versioning isn't. Fortunately, the
needs across all domains are very similar. We all have to support multiple weights, handle model code changes, strive 
for reproducibility etc. Standardizing the way we do this across domains is important because:
1. Code changes affecting the models are common and part of the standard development cycle of all domains.
2. Updating the model weights with more accurate ones trained using new recipes is a common problem 
   [ [5](https://github.com/pytorch/vision/issues/3995) ]. Same applies to providing concurrently multiple versions of 
   the weights trained on different Datasets (to cover for different taxonomies, languages etc).
3. Providing an aligned API will improve user experience while combining the Domain libraries in their training recipes.

## Previous work

This is not the first time the model versioning problem arises [ [6](https://github.com/pytorch/vision/issues/2955) ] 
as it has previously been discussed by the domain library maintainers. 

In TorchVision, model versioning related issues have been handled in a case-by-case basis 
[ [7](https://github.com/pytorch/vision/issues/2599), [8](https://github.com/pytorch/vision/issues/2326), 
[9](https://github.com/pytorch/vision/pull/3205), [10](https://github.com/pytorch/vision/pull/1224) ]. We typically
try to maintain BC as much as possible except in cases where the issue is considered a bug or extremely detrimental to
the user experience. In the latter cases, we often deploy BC-breaking fixes but try to reduce the effects of the change
as much as possible [ [11](https://github.com/pytorch/vision/pull/2940) ]. TorchAudio has only recently introduced the
pre-trained models and their API is currently in beta while TorchText is currently working on introducing its first
pre-trained models so their solution is in prototype phase. Finally, though PyTorch Core doesn't provide pre-trained
models, occasionally there is the need of handling code changes on existing Layers and operators. These are typically
handled with a mix of version parameters, deprecation warnings and method renamings
[ [12](https://github.com/pytorch/pytorch/blob/294db060/torch/nn/quantized/dynamic/modules/linear.py#L44-L49),
[13](https://github.com/pytorch/pytorch/blob/c371542e/caffe2/python/cnn.py#L182-L183),
[14](https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/caffe2/python/brew.py#L65-L66) ]

## Proposal

This repository aims to serve as a live RFC document capable of show-casing the proposed API and utilities, providing 
examples of how to address the most common model-versioning scenarios and offering actual implementations for some of 
the real-world models included in the Domain libraries.

### Specifications

The proposed solution should meet the following criteria:
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
   registration mechanisms etc) but these are out of scope for this document. Note that when possible, we illustrate
   with code examples that our solution is compatible with such mechanisms, but they are not part of our proposal.
2. We focus on the public API and not on implementation details. Though the repo contains private helper methods used to
   [construct models](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/resnet.py#L19-L43),
   these are not part of the proposal and libraries can adapt them to their needs and current practices.

### Detailed design

#### Proposed: Separate model builder and weight parameter for each code version

We propose maintaining the existing model builder methods that all Domain libraries support to construct models and 
use Enum data classes to pass information about the pre-trained weights. Each model variant will have its own method
and weights. When BC-breaking changes are introduced we will introduce new methods. Example:

```python
@dataclass
class ResNet50V2Weights(Enum):
    ImageNet1K_RefV1 = (
        # attribute 1,
        # ...
        # attribute n,
    )

resnet50_v2(weights=ResNet50V2Weights.ImageNet1K_RefV1)
```

The above approach:
- Is fully BC and easy to adopt for all domain libraries.
- Covers all mandatory criteria listed above.
- Supports all the nice-to-haves.

#### Deep Dive

Instead of providing code snippets in this README, we link directly to the actual implementations and code examples 
where we document everything extensively.

The best way to see how the proposal works is to check the `examples` folder, where we focus on 3 Model Versioning
scenarios that we had to address in the past:
1. [Same code, different model weights (BC)](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario1.py)
2. [Updated default Hyper-params (BC-breaking)](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario2.py)
3. [Code change which affects the model behaviour but architecture remains the same (BC-breaking)](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario3.py)

Our proposal consists of the following key components:
- The [Weights](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L17-L71) data class
  which stores crucial information about the pre-trained weights.
- The [model builder methods](https://github.com/datumbox/dapi-model-versioning/blob/main/examples/scenario1.py#L77-L96)
  which construct the model variants and load the pre-trained weights.

We also offer two optional components:
- The [ContetParams](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L74-L132) class
  which allows us to minimize the effects of BC-breaking changes to classes such as Layers and Modules.
- A [Registration](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L135-L220)
  mechanism similar to the one adapted for the prototype datasets on 
  [TorchVision](https://github.com/pytorch/vision/blob/main/torchvision/prototype/datasets/_api.py#L13) to show-case
  that our proposal is compatible with it.

### Alternatives considered

Here we briefly list the alternatives that we considered along with some of the reasons we didn't select them. Note that
in all cases, we consider using Enums instead of strings. To read more on why check this 
[section](https://github.com/datumbox/dapi-model-versioning/blob/main/dapi_lib/models/_api.py#L43-L47).

#### Alt 1: Single model builder and weight parameter for all code versions

```python
resnet50(weights=ResNet50Weights.V2_ImageNet1K_RefV1)
```

Pros:
- Small number of model builders and Enum classes to maintain.

Cons:
- All versions must be handled in a single method leading to complex implementations.
- Harder to document and unit-test using standard python tools.
- Since the version is linked to the weight enum, it would require the introduction of special enums to denote that no
  pre-trained weights should be loaded.

#### Alt 2: Single model builder, two different arguments to denote the version and the weights

```python
resnet50(version=2, weights=ResNet50Weights.ImageNet1K_RefV1)
```

Pros:
- Small number of model builders and Enum classes to maintain.
- No longer requires special enums.

Cons:
- All versions must be handled in a single method leading to complex implementations.
- Harder to document and unit-test using standard python tools.
- Difficult for the users to tell which `version` goes with what `weights` enum.


#### Alt 3: Separate model builder for each code version and weights combination

```python
resnet50_v2_imagenet_ref1(pretrained=True)
```

Pros:
- No changes required. Keeps things as they are in domain libraries.
- Keeps things immutable and frozen in time.

Cons:
- The number of methods increases multiplicatively with each version, dataset, recipe combination.
- Can lead to a lot of legacy code.
- It's a nonsolution. It does not really addresses the Versioning problem.

### Repository Structure

Below we describe the structure of the repository:

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
  looks from the user purspective:
    - The `image_classification.py` and `image_detection.py` show-case the new API on Vision.
    - The `test_encoding.py` gives us an example on how Text could structure its models. Note that because TorchText
    currently doesn't provide pre-trained models on the public repo, we use huggingface's Roberta.
    - The `text_to_speech.py` provides an example of implementing the new API on Audio.
- Other details:
    - The `assets` folder contains a couple of assets necessary for the demos.
    - An `output` folder will be created on the root of the project after running the demos.
    - The `third_party` package contains copy-pasted code necessary for the demos and it is not part of this RFC.
    - The `requirements.txt` file contains a list of all dependencies for running the code on this repo.

## Next steps

- [ ] Share the Repo with the leads of TorchAudio and TorchText and adapt it based on their feedback. 
- [ ] Present the proposal to the broad DAPI maintainers and make amendments based on their input. 
- [ ] Make the Repo public, collect feedback from the community and iterate on the proposal.
- [ ] Implement the proposal on domain libraries.
