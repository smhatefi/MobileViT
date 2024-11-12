import os

# Use Either one of these as backend.
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_BACKEND"] = "torch"


import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.load_weights import return_models
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluate the model on sample images')

parser.add_argument('--model_size', type=str, default='XS',
                    help='Model Size = S, XS or XXS')

parser.add_argument('--image_path', type=str, default='res/cat.jpg',
                    help='Full path to the image file')

args = parser.parse_args()
model_size = args.model_size
model  = return_models(model_size)

# Load the labels (ImageNet class names)
with open("res/imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def test_prediction(*, image_path, model=None, image_shape=(256, 256), show=True):
    # Load and process the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR) # NOT CONVERTED to RGB (required).

    img = cv2.resize(img, image_shape)

    if show:
        plt.imshow(img[:, :, ::-1])

    img = img  / 255. # Normalize pixel values to [0, 1]
    img = img.astype("float32")  # Ensure the correct type for TensorFlow
    # Add the batch dimension
    img = np.expand_dims(img, 0)  # Shape becomes (1, 256, 256, 3)

    # Perform prediction
    preds = model.predict(img, verbose=0)

    # Output prediction
    print(f"Model: {model.name}, Predictions: {labels[preds.argmax()]}")


test_prediction(image_path=args.image_path, model=model, show=True)