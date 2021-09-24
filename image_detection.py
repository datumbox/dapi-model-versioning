from pathlib import Path
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from dapi_lib import models


img = read_image("assets/puppies.jpg")

# Initialize model, weights are optional
weights = models.FasterRCNNResNet50FPNWeights.Coco_RefV1
model = models.fasterrcnn_resnet50_fpn(weights=weights)
# model, weights = models.get('fasterrcnn_resnet50_fpn')

model.eval()

# Initialize inference transforms
preprocess = weights.transforms()

# Apply inference presets
image = preprocess(img)

# Make predictions
prediction = model([image])[0]

# Use meta to get labels
labels = [weights.meta['classes'][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction['boxes'],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
Path("./output").mkdir(parents=True, exist_ok=True)
im.save("./output/object-detection.jpg")
im.show()
