import torch
import requests
import gradio as gr
from PIL import Image
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(inp):
 inp = preprocess(inp).unsqueeze(0)
 with torch.no_grad():
  prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
 return confidences

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    examples=["content/lion.jpg", "content/cheetah.jpg"]
)

demo.launch(server_name="127.0.0.1", server_port= 7860)
