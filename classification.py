from torchvision.io.image import read_image

from dapi_lib import models


img = read_image("assets/puppies.jpg")

# Initialize model, weights are optional
weights = models.ResNet50Weights.ImageNet1K_RefV1
model = models.resnet50(weights=weights)

model.eval()

# Transforms need to be initialized when needed because they might have memory
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
