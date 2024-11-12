import os

# Use Either one of these as backend.
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from utils.load_weights import return_models
from utils.dataset import downloadImageNet
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up argument parser
parser = argparse.ArgumentParser(description='Test the model on ImageNet')

parser.add_argument('--model_size', type=str, default='XS',
                    help='Model Size = S, XS or XXS')

args = parser.parse_args()
model_size = args.model_size
print('Model Size:',model_size)
model  = return_models(model_size)
model = model.to(device)


# Define transformation and dataset
transform = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.ToTensor()
])

downloadImageNet()
val_dataset = ImageNet(root='./res', split='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)


# Top-1 Accuracy
def evaluate_top1_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    i = 0
    print("Evaluating Top-1 Accuracy...")
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            images = images.permute(0, 2, 3, 1)  # Convert to (batch, height, width, channels)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Batch {i+1}/{len(dataloader)}")
            i += 1
    return correct / total

top1_accuracy = evaluate_top1_accuracy(model, val_loader)
print(f'Top-1 Accuracy: {top1_accuracy:.2%}')


# Showing model parameters
num_params = model.count_params()
print(f'Number of Parameters: {num_params}')


# FLOPs
os.system("pip install torchprofile")

from torchprofile import profile_macs

model.eval()
input_tensor = torch.randn(1, 3, 256, 256).to(device)
input_tensor = input_tensor.permute(0, 2, 3, 1)
# Calculate MACs (multiply-accumulate operations)
macs = profile_macs(model, input_tensor)
flops = 2 * macs  # FLOPs are generally twice the MACs in CNNs
print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")


# Inference Time
import time

def measure_inference_time_keras(model, dataloader, num_batches=100):
    start_time = time.time()
    
    for i, (images, _) in enumerate(dataloader):
        if i == num_batches:
            break
        
        # Rearrange dimensions to match Keras expected input shape (batch_size, height, width, channels)
        images = images.permute(0, 2, 3, 1).to(device)
        
        with torch.no_grad():  # Ensure no gradients are calculated
            _ = model(images)  # Run the inference
    
    avg_inference_time = (time.time() - start_time) / num_batches
    return avg_inference_time

avg_inference_time = measure_inference_time_keras(model, val_loader)
print(f'Average Inference Time per Batch: {avg_inference_time:.4f} seconds')
