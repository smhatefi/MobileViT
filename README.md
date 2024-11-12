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
- `utils/dataset.py`: Downloads ImageNet val dataset for validating th model.

## Usage

### Evaluating the model
For evaluating the model on the example images (loacted in the `res/` directory) run the `evaluate.py` script:
```
python evaluate.py
```
This will use pre-trained weights to evaluate the model.

### Testing the model
For validating the model on ImageNet val set run the `test.py` script:
```
python test.py
```
This will use pre-trained weights to test the model on ImageNet-1k val dataset.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The MobileViT model architecture is inspired by the original [MobileViT paper](https://arxiv.org/abs/2110.02178).
- The code is heavily borrowed from [this amazing post on LearnOpenCV website](https://learnopencv.com/mobilevit-keras-3/).
