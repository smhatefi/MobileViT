from typing import Union, Optional
import keras.ops as kops
from keras import Model, Input
from keras.layers import (Layer, Dense, Dropout, GlobalAveragePooling2D, LayerNormalization, Concatenate)

from utils.layers import ConvLayer, InvertedResidualBlock, Transformer
from configs import get_mobile_vit_v1_configs


class MobileViT_v1_Block(Layer):
    def __init__(
        self,
        out_filters: int = 64,
        embedding_dim: int = 90,
        patch_size: Union[int, tuple] = 2,
        transformer_repeats: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_drop: float = 0.0,
        linear_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.out_filters = out_filters
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.transformer_repeats = transformer_repeats
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_drop = attention_drop
        self.linear_drop = linear_drop

        self.patch_size_h, self.patch_size_w = patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        self.patch_size_h, self.patch_size_w = kops.cast(self.patch_size_h, dtype="int32"), kops.cast(self.patch_size_w, dtype="int32")

        # # local_feature_extractor 1 and 2
        self.local_rep_layer_1 = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)
        self.local_rep_layer_2 = ConvLayer(num_filters=self.embedding_dim, kernel_size=1, strides=1, use_bn=False, use_activation=False, use_bias=False)

        self.transformer_layers = [
            Transformer(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                attention_drop=self.attention_drop,
                linear_drop=self.linear_drop,
            )
            for _ in range(self.transformer_repeats)
        ]

        self.transformer_layer_norm = LayerNormalization(epsilon=1e-5)

        # Fusion blocks
        self.local_features_3 = ConvLayer(num_filters=self.out_filters, kernel_size=1, strides=1, use_bn=True, use_activation=True)
        self.concat = Concatenate(axis=-1)
        self.fuse_local_global = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):

        fmH, fmW = kops.shape(x)[1], kops.shape(x)[2]

        local_representation = self.local_rep_layer_1(x)
        local_representation = self.local_rep_layer_2(local_representation)
        out_channels = local_representation.shape[-1]

        # Transformer as Convolution Steps
        # --------------------------------
        # # Unfolding

        unfolded, info_dict = self.unfolding(local_representation)

        # # Infomation sharing/mixing --> global representation
        for layer in self.transformer_layers:
            unfolded = layer(unfolded)

        global_representation = self.transformer_layer_norm(unfolded)

        # #Folding
        folded = self.folding(global_representation, info_dict=info_dict, outH=fmH, outW=fmW, outC=out_channels)

        # Fusion
        local_mix = self.local_features_3(folded)
        fusion = self.concat([x, local_mix])
        fusion = self.fuse_local_global(fusion)

        return fusion

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_filters": self.out_filters,
                "embedding_dim": self.embedding_dim,
                "patch_size": self.patch_size,
                "transformer_repeats": self.transformer_repeats,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "attention_drop": self.attention_drop,
                "linear_drop": self.linear_drop,
            }
        )
        return config

    def unfolding(self, feature_map):
        # Initially convert channel-last to channel-first for processing
        shape = kops.shape(feature_map)
        batch_size, orig_h, orig_w, in_channels = shape[0], shape[1], shape[2], shape[3]
        feature_map = kops.transpose(feature_map, [0, 3, 1, 2])  # [B, H, W, C] -> [B, C, H, W]

        patch_area = self.patch_size_w * self.patch_size_h

        orig_h, orig_w = kops.cast(orig_h, dtype="int32"), kops.cast(orig_w, dtype="int32")

        h_ceil = kops.ceil(orig_h / self.patch_size_h)
        w_ceil = kops.ceil(orig_w / self.patch_size_w)

        new_h = kops.cast(h_ceil * kops.cast(self.patch_size_h, dtype=h_ceil.dtype), dtype="int32")
        new_w = kops.cast(w_ceil * kops.cast(self.patch_size_w, dtype=h_ceil.dtype), dtype="int32")

        # Condition to decide if resizing is necessary
        resize_required = kops.logical_or(kops.not_equal(new_w, orig_w), kops.not_equal(new_h, orig_h))
        feature_map = kops.cond(
            resize_required,
            true_fn=lambda: kops.image.resize(feature_map, [new_h, new_w], data_format="channels_first"),
            false_fn=lambda: feature_map,
        )

        num_patch_h = new_h // self.patch_size_h
        num_patch_w = new_w // self.patch_size_w
        num_patches = num_patch_h * num_patch_w

        # Handle dynamic shape multiplication
        dynamic_shape_mul = kops.prod([batch_size, in_channels * num_patch_h])

        # Reshape and transpose to create patches
        reshaped_fm = kops.reshape(feature_map, [dynamic_shape_mul, self.patch_size_h, num_patch_w, self.patch_size_w])
        transposed_fm = kops.transpose(reshaped_fm, [0, 2, 1, 3])
        reshaped_fm = kops.reshape(transposed_fm, [batch_size, in_channels, num_patches, patch_area])
        transposed_fm = kops.transpose(reshaped_fm, [0, 3, 2, 1])
        patches = kops.reshape(transposed_fm, [batch_size * patch_area, num_patches, in_channels])

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": resize_required,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
            "patch_area": patch_area,
        }

        return patches, info_dict

    def folding(self, patches, info_dict, outH, outW, outC):
        # Ensure the input patches tensor has the correct dimensions
        assert len(patches.shape) == 3, f"Tensor should be of shape BPxNxC. Got: {patches.shape}"

        # Reshape to [B, P, N, C]
        patches = kops.reshape(patches, [info_dict["batch_size"], info_dict["patch_area"], info_dict["total_patches"], -1])

        # Get shape parameters for further processing
        shape = kops.shape(patches)
        batch_size = shape[0]
        channels = shape[3]

        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # Transpose dimensions [B, P, N, C] --> [B, C, N, P]
        patches = kops.transpose(patches, [0, 3, 2, 1])

        # Calculate total elements dynamically
        num_total_elements = batch_size * channels * num_patch_h

        # Reshape to match the size of the feature map before splitting into patches
        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = kops.reshape(patches, [num_total_elements, num_patch_w, self.patch_size_h, self.patch_size_w])

        # Transpose to switch width and height axes [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = kops.transpose(feature_map, [0, 2, 1, 3])

        # Reshape back to the original image dimensions [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        # Reshape back to [B, C, H, W]
        new_height = num_patch_h * self.patch_size_h
        new_width = num_patch_w * self.patch_size_w
        feature_map = kops.reshape(feature_map, [batch_size, -1, new_height, new_width])

        # Conditional resizing using kops.cond
        feature_map = kops.cond(
            info_dict["interpolate"],
            lambda: kops.image.resize(feature_map, info_dict["orig_size"], data_format="channels_first"),
            lambda: feature_map,
        )

        feature_map = kops.transpose(feature_map, [0, 2, 3, 1])
        feature_map = kops.reshape(feature_map, (batch_size, outH, outW, outC))

        return feature_map


# Building MobileViT Architecture
def MobileViT_v1(
    configs,
    dropout: float = 0.1,
    linear_drop: float = 0.0,
    attention_drop: float = 0.0,
    num_classes: int | None = 1000,
    input_shape: tuple[int, int, int] = (256, 256, 3),
    model_name: str = f"MobileViT_v1-S",
):
    """
    Arguments
    --------

        configs: A dataclass instance with model information such as per layer output channels, transformer embedding dimensions, transformer repeats, IR expansion factor

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

        model_type: (str)   Model to create

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    input_layer = Input(shape=input_shape)

    # Block 1
    out = ConvLayer(
        num_filters=configs.block_1_1_dims,
        kernel_size=3,
        strides=2,
        name="block-1-Conv",
    )(input_layer)

    out = InvertedResidualBlock(
        in_channels=configs.block_1_1_dims,
        out_channels=configs.block_1_2_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-1-IR2",
    )(out)

    # Block 2
    out = InvertedResidualBlock(
        in_channels=configs.block_1_2_dims,
        out_channels=configs.block_2_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR1",
    )(out)

    out = InvertedResidualBlock(
        in_channels=configs.block_2_1_dims,
        out_channels=configs.block_2_2_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR2",
    )(out)

    out = InvertedResidualBlock(
        in_channels=configs.block_2_2_dims,
        out_channels=configs.block_2_3_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR3",
    )(out)

    # Block 3
    out = InvertedResidualBlock(
        in_channels=configs.block_2_2_dims,
        out_channels=configs.block_3_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViT_v1_Block(
        out_filters=configs.block_3_2_dims,
        embedding_dim=configs.tf_block_3_dims,
        transformer_repeats=configs.tf_block_3_repeats,
        name="MobileViTBlock-1",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 4
    out = InvertedResidualBlock(
        in_channels=configs.block_3_2_dims,
        out_channels=configs.block_4_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-4-IR1",
    )(out)

    out = MobileViT_v1_Block(
        out_filters=configs.block_4_2_dims,
        embedding_dim=configs.tf_block_4_dims,
        transformer_repeats=configs.tf_block_4_repeats,
        name="MobileViTBlock-2",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 5
    out = InvertedResidualBlock(
        in_channels=configs.block_4_2_dims,
        out_channels=configs.block_5_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-5-IR1",
    )(out)

    out = MobileViT_v1_Block(
        out_filters=configs.block_5_2_dims,
        embedding_dim=configs.tf_block_5_dims,
        transformer_repeats=configs.tf_block_5_repeats,
        name="MobileViTBlock-3",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    out = ConvLayer(num_filters=configs.final_conv_dims, kernel_size=1, strides=1, name="final_conv")(out)

    if num_classes:
        # Output layer
        out = GlobalAveragePooling2D()(out)

        if linear_drop > 0.0:
            out = Dropout(rate=dropout)(out)

        out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def build_MobileViT_v1(
    model_type: str = "S",
    num_classes: int = 1000,
    input_shape: tuple = (256, 256, 3),
    include_top: bool = True,  # Whether to include the classification layer in the model
    updates: Optional[dict] = None,
    **kwargs,
):
    """
    Create MobileViT-v1 Classification models or feature extractors with optional pretrained weights.

    Arguments:
    ---------
        model_type: (str)   MobileViT version to create. Options: S, XS, XXS
        num_classes: (int)   Number of output classes
        input_shape: (tuple) Input shape -> H, W, C
        include_top: (bool) Whether to include the classification layers
        updates: (dict) a key-value pair indicating the changes to be made to the base model.

    Additional arguments:
    ---------------------
        linear_drop: (float) Dropout rate for Dense layers
        attention_drop: (float) Dropout rate for the attention matrix
    """
    model_type = model_type.upper()
    if model_type not in ("S", "XS", "XXS"):
        raise ValueError("Bad Input. 'model_type' should be one of ['S', 'XS', 'XXS']")

    updated_configs = get_mobile_vit_v1_configs(model_type, updates=updates)

    # Build the base model
    model = MobileViT_v1(
        configs=updated_configs,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name=f"MobileViT_v1-{model_type}",
        **kwargs,
    )

    return model