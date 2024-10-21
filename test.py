import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.load_weights import return_models


keras_model_XXS = return_models("XXS")
keras_model_XS  = return_models("XS")
keras_model_S   = return_models("S")

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


cat_image_path = "res/cat.jpg"

test_prediction(image_path=cat_image_path, model=keras_model_XXS, show=True)
test_prediction(image_path=cat_image_path, model=keras_model_XS)
test_prediction(image_path=cat_image_path, model=keras_model_S)

panda_image_path = "res/panda.JPG"
test_prediction(image_path=panda_image_path, model=keras_model_XXS, show=True)
test_prediction(image_path=panda_image_path, model=keras_model_XS)
test_prediction(image_path=panda_image_path, model=keras_model_S)