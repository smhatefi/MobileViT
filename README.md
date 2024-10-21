# MobileViT
This repository contains the implementation of MobileViT model in Keras 3.

## Project Structure
- `mobilevit.py`: Contains the main model implementation.
- `configs.py`: Contains the model configurations.
- `test.py`: Script for evaluating the model.
- `main.py`: Main script for running the model.
- `res/`: Contains the resources for running scripts.
- `utils/layers.py`: Contains base layers.
- `utils/load_weights.py`: Contains utility functions for porting weights from the official MobileViT model.

## Usage

For evaluating the model on the example images (loacted in the `res/` directory) run the `main.py` script:
```
python main.py
```

This will sets the keras3 backend and calls the `test.py`.
`test.py` itself uses `utils/load_weights.py` to port the official weights of MobileViT to our model.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The MobileViT model architecture is inspired by the original [MobileViT paper](https://arxiv.org/abs/2110.02178).
- The code is heavily borrowed from [this amazing post on LearnOpenCV website](https://learnopencv.com/mobilevit-keras-3/).
