import urllib
import torch
import torch.nn as nn
import torchvision
import json
from torchvision import transforms
from PIL import Image
import coremltools as ct

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


# Load the model (deeplabv3)
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()

# Load a sample image (cat_dog.jpg)
input_image = Image.open("cat_dog.jpg")
input_image.show()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]
torch_predictions = output.argmax(0)


def display_segmentation(input_image, output_predictions):
    # Create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(
        output_predictions.byte().cpu().numpy()
    ).resize(input_image.size)
    r.putpalette(colors)

    # Overlay the segmentation mask on the original image
    alpha_image = input_image.copy()
    alpha_image.putalpha(255)
    r = r.convert("RGBA")
    r.putalpha(128)
    seg_image = Image.alpha_composite(alpha_image, r)
    # display(seg_image) -- doesn't work
    seg_image.show()


display_segmentation(input_image, torch_predictions)


# Wrap the Model to Allow Tracing*
class WrappedDeeplabv3Resnet101(nn.Module):

    def __init__(self):
        super(WrappedDeeplabv3Resnet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()

    def forward(self, x):
        res = self.model(x)
        x = res["out"]
        return x


# Trace the Wrapped Model
traceable_model = WrappedDeeplabv3Resnet101().eval()
trace = torch.jit.trace(traceable_model, input_batch)

# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

# Save the model without new metadata
mlmodel.save("SegmentationModel_no_metadata.mlmodel")

# Load the saved model
mlmodel = ct.models.MLModel("SegmentationModel_no_metadata.mlmodel")

# Add new metadata for preview in Xcode
labels_json = {
    "labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train",
               "tvOrMonitor"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save("SegmentationModel_with_metadata.mlmodel")