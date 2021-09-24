from PIL import Image

from dapi_lib import models


img = Image.open("assets/puppies.jpg")

# Initialize model, weights are optional
weights = models.ResNet50Weights.ImageNet1K_RefV1
model = models.resnet50(weights=weights)
# model, weights = models.get('resnext101_32x8d')

model.eval()

# Initialize inference transforms
preprocess = weights.transforms()

# Apply inference presets
batch = preprocess(img).unsqueeze(0)

# Make predictions
prediction = model(batch).squeeze(0).softmax(0)
label = prediction.argmax().item()
score = prediction[label].item()

# Use meta to get label
class_name = weights.meta['classes'][label]
print(class_name, score)
