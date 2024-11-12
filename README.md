# MobileViT
This repository contains the implementation of MobileViT model in Keras 3.

## Project Structure
- `mobilevit.py`: Contains the main model implementation.
- `configs.py`: Contains the model configurations.
- `evaluate.py`: Script for evaluating the model.
- `test.py`: Script for validating the model on ImageNet val set.
- `res/`: Contains the resources for running scripts.
- `utils/layers.py`: Contains base layers.
- `utils/load_weights.py`: Contains utility functions for porting weights from the official MobileViT model.
- `utils/dataset.py`: Downloads ImageNet val dataset for validating the model.

## Usage
You can evaluate the model on sample images or test the model on ImageNet-1k val dataset.
In both cases the pre-trained weights from the official MobileViT model are used.

### Evaluating the model
For evaluating the model on the example images (loacted in the `res/` directory) run the `evaluate.py` script:
```
python evaluate.py --model_size S --image_path res/panda.JPG
```
You should set the model size (S, XS, or XXS) and full image path as arguments.

### Testing the model
For validating the model on ImageNet val set run the `test.py` script:
```
python test.py --model_size XS
```
You can set the model size (S, XS, or XXS).

## Results
I tested all of the models (S, XS and XXS) on **T4 GPU of Google colab** and here is the obtained results:

| Model size | Top-1 Accuracy | Number of Parameters | FLOPs | Inference Time per Batch |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| XXS | 49.83% | 1.27 M | 0.44 G | 0.2756 seconds |
| XS | 58.53% | 2.32 M | 0.86 G | 0.2812 seconds |
| S | 63.50% | 5.59 M | 1.85 G | 0.2858 seconds |

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The MobileViT model architecture is inspired by the original [MobileViT paper](https://arxiv.org/abs/2110.02178).
- The code is heavily borrowed from [this amazing post on LearnOpenCV website](https://learnopencv.com/mobilevit-keras-3/).
