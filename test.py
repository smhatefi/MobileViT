import os

# Use Either one of these as backend.
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.load_weights import return_models
from utils.dataset import downloadImageNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

keras_model_XS  = return_models("XS")
keras_model_XS = keras_model_XS.to(device)


# Define transformation and dataset
transform = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

downloadImageNet()
val_dataset = ImageNet(root='./res', split='val', transform=transform)
#val_dataset = datasets.ImageFolder('res/imagenet/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)


# Top-1 Accuracy
def evaluate_top1_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

top1_accuracy = evaluate_top1_accuracy(keras_model_XS, val_loader)
print(f'Top-1 Accuracy: {top1_accuracy:.2%}')


# Showing model parameters
num_params = keras_model_XS.count_params()
print(f'Number of Parameters: {num_params}')


os.system("pip install keras-flops")

from keras_flops import get_flops

flops = get_flops(keras_model_XS, batch_size=1)
print(f"FLOPs: {flops / 10**9:.2f} GigaFLOPs")


# Inference Time
import time

def measure_inference_time_keras(model, dataloader, num_batches=100):
    start_time = time.time()
    for i, (images, _) in enumerate(dataloader):
        if i == num_batches:
            break
        _ = model(images)  # Run the inference
    avg_inference_time = (time.time() - start_time) / num_batches
    return avg_inference_time

avg_inference_time = measure_inference_time_keras(keras_model_XS, val_loader)
print(f'Average Inference Time per Batch: {avg_inference_time:.4f} seconds')
