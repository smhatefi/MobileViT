import torch
from mobilevit import build_MobileViT_v1
import os


class WeightsLayerIterator:
    def __init__(self, pytorch_weights, keras_model):
        self.keras_model = keras_model
        self.pytorch_weights = pytorch_weights
        self.keras_layer_is_depthwise = False
        self.keras_layer_is_einsum = False

    def get_next_pytorch_weight(self):
        count = 0
        for idx, (param_name, param) in enumerate(self.pytorch_weights.items()):

            sentence = "{count} {param_name} ----> {param_shape}"
            if "num_batches_tracked" in param_name:
                continue
            if "conv.weight" in param_name:
                if self.keras_layer_is_depthwise:
                    param = param.permute(2, 3, 0, 1)
                else:
                    param = param.permute(2, 3, 1, 0)

            elif len(param.shape) == 2:
                param = param.T
            count += 1
            yield sentence.format(count=count, param_name=param_name, param_shape=param.shape), param

    def get_keras_weight(self):
        count = 0
        for idx, param in enumerate(self.keras_model.variables):
            if "seed_generator_state" in param.path:
                continue

            self.keras_layer_is_depthwise = True if "depthwise_conv2d" in param.path else False

            count += 1
            yield f"{count} {param.path} ----> {param.shape}", param

def get_pytorch2keras_layer_weights_mapping(pytorch_weights, keras_model):
    wl = WeightsLayerIterator(pytorch_weights=pytorch_weights, keras_model=keras_model)
    layer_mapping = {}
    for (keras_sentence, keras_weight), (pytorch_sentence, pytorch_weight) in zip(wl.get_keras_weight(), wl.get_next_pytorch_weight()):
        keras_layer_name = keras_weight.path.split("/")[0]
        layer_mapping.setdefault(keras_layer_name, list())
        layer_mapping[keras_layer_name].append(pytorch_weight.numpy())

    return layer_mapping


def load_weights_in_keras_model(keras_model, layer_mapping):
    for keras_layer in keras_model.layers[1:]:
        keras_layer_name = keras_layer.name

        if "global_average_pooling2d" in keras_layer_name:
            continue

        if "dropout" in keras_layer.name:
            continue

        from_pt = list(layer_mapping[keras_layer_name])
        keras_model.get_layer(keras_layer_name).set_weights(from_pt)
    return keras_model


def return_models(model_type: str = "S"):
    if model_type == "S":
        # Download pretrained pytorch model weight
        os.system("wget -q -O mobilevit_s.pt https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt")
        pytorch_weights_s   =  torch.load("mobilevit_s.pt",   map_location="cpu")
        keras_model_S   = build_MobileViT_v1(model_type="S",   input_shape=(256, 256, 3))
        keras_model_s_layer_mapping   = get_pytorch2keras_layer_weights_mapping(pytorch_weights=pytorch_weights_s, keras_model=keras_model_S)
        keras_model_S   = load_weights_in_keras_model(keras_model=keras_model_S, layer_mapping=keras_model_s_layer_mapping)
        return keras_model_S
    
    elif model_type == "XS":
        os.system("wget -q -O mobilevit_xs.pt https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt")
        pytorch_weights_xs  =  torch.load("mobilevit_xs.pt",  map_location="cpu")
        keras_model_XS  = build_MobileViT_v1(model_type="XS",  input_shape=(256, 256, 3))
        keras_model_xs_layer_mapping  = get_pytorch2keras_layer_weights_mapping(pytorch_weights=pytorch_weights_xs, keras_model=keras_model_XS)
        keras_model_XS  = load_weights_in_keras_model(keras_model=keras_model_XS, layer_mapping=keras_model_xs_layer_mapping)
        return keras_model_XS
    
    elif model_type == "XXS":
        os.system("wget -q -O mobilevit_xxs.pt https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt")
        pytorch_weights_xxs =  torch.load("mobilevit_xxs.pt", map_location="cpu")
        keras_model_XXS = build_MobileViT_v1(model_type="XXS", input_shape=(256, 256, 3))
        keras_model_xxs_layer_mapping = get_pytorch2keras_layer_weights_mapping(pytorch_weights=pytorch_weights_xxs, keras_model=keras_model_XXS)
        keras_model_XXS = load_weights_in_keras_model(keras_model=keras_model_XXS, layer_mapping=keras_model_xxs_layer_mapping)
        return keras_model_XXS
    
    else:
        return None