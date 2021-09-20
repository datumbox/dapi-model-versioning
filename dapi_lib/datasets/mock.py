# This a mock object and is going to be replaced with a mechanism provided by
# https://github.com/pmeier/torchvision-datasets-rework/

with open("./assets/imagenet_classes.txt", "r") as f:
    imagenet_classes = [s.strip() for s in f.readlines()]


class ImageNet:
    classes = imagenet_classes
