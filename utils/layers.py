from typing import Union, Optional
import keras.ops as kops
from keras.layers import (
    Layer,
    Conv2D,
    DepthwiseConv2D,
    BatchNormalization,
    Activation,
    Dense,
    Dropout,
    Softmax,
    LayerNormalization,
    Identity,
    ZeroPadding2D,
)


# https://www.tensorflow.org/guide/mixed_precision#ensuring_gpu_tensor_cores_are_used
def make_divisible(v: Union[int, float], divisor: Optional[Union[int, float]] = 8, min_value: Optional[Union[int, float]] = None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvLayer(Layer):
    def __init__(
        self,
        num_filters: int = 16,
        kernel_size: int = 3,
        strides: int = 2,
        use_activation: bool = True,
        use_bn: bool = True,
        use_bias: bool = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bn = use_bn
        self.use_activation = use_activation
        self.use_bias = use_bias if use_bias is not None else (False if self.use_bn else True)

        if self.strides == 2:
            self.zero_pad = ZeroPadding2D(padding=(1, 1))
            conv_padding = "valid"
        else:
            self.zero_pad = Identity()
            conv_padding = "same"
        self.conv = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, strides=self.strides, padding=conv_padding, use_bias=self.use_bias)

        if self.use_bn:
            self.bn = BatchNormalization(epsilon=1e-05, momentum=0.1)

        if self.use_activation:
            self.activation = Activation("swish")

    def call(self, x, **kwargs):
        x = self.zero_pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        if self.use_activation:
            x = self.activation(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "use_activation": self.use_activation,
                "use_bn": self.use_bn,
            }
        )
        return config


class InvertedResidualBlock(Layer):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        depthwise_stride: int = 1,
        expansion_factor: Union[int, float] = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input Parameters

        self.num_in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_stride = depthwise_stride
        self.expansion_factor = expansion_factor

        num_out_channels = int(make_divisible(self.out_channels, divisor=8))
        expansion_channels = int(make_divisible(self.expansion_factor * self.num_in_channels))

        # Layer Attributes
        apply_expansion = expansion_channels > self.num_in_channels
        self.residual_connection = True if (self.num_in_channels == num_out_channels) and (self.depthwise_stride == 1) else False

        # Layers
        if apply_expansion:
            self.expansion_conv_block = ConvLayer(num_filters=expansion_channels, kernel_size=1, strides=1, use_activation=True, use_bn=True)
        else:
            self.expansion_conv_block = Identity()

        self.depthwise_conv_zero_pad = ZeroPadding2D(padding=(1, 1))
        self.depthwise_conv = DepthwiseConv2D(kernel_size=3, strides=self.depthwise_stride, padding="valid", use_bias=False)
        self.bn = BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.activation = Activation("swish")
        self.out_conv_block = ConvLayer(num_filters=num_out_channels, kernel_size=1, strides=1, use_activation=False, use_bn=True)

    def call(self, data, **kwargs):
        out = self.expansion_conv_block(data)
        out = self.depthwise_conv_zero_pad(out)
        out = self.depthwise_conv(out)
        out = self.bn(out)
        out = self.activation(out)
        out = self.out_conv_block(out)

        if self.residual_connection:
            return out + data

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.num_in_channels,
                "out_channels": self.out_channels,
                "depthwise_stride": self.depthwise_stride,
                "expansion_factor": self.expansion_factor,
            }
        )
        return config
    

class MHSA(Layer):
    def __init__(
        self,
        num_heads: int = 2,
        embedding_dim: int = 64,
        projection_dim: int = None,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim if projection_dim else embedding_dim // num_heads
        self.qkv_bias = qkv_bias
        self.scale = self.projection_dim**-0.5

        self.qkv = Dense(3 * self.num_heads * self.projection_dim, use_bias=qkv_bias)
        self.proj = Dense(embedding_dim, use_bias=qkv_bias)
        self.attn_dropout = Dropout(attention_drop)
        self.linear_dropout = Dropout(linear_drop)
        self.softmax = Softmax()

    def build(self, input_shape):
        # You can perform setup tasks that depend on the input shape here
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, projection_dim)
        x = kops.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # Transpose to shape (batch_size, num_heads, seq_len, projection_dim)
        return kops.transpose(x, axes=(0, 2, 1, 3))

    def call(self, x):
        batch_size = kops.shape(x)[0]

        # Project and reshape to (batch_size, seq_len, 3, num_heads, projection_dim)
        qkv = self.qkv(x)
        qkv = kops.reshape(qkv, (batch_size, -1, 3, self.num_heads, self.projection_dim))
        qkv = kops.transpose(qkv, axes=(0, 2, 1, 3, 4))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        q *= self.scale

        # Attention mechanism
        attn_logits = kops.matmul(q, kops.transpose(k, axes=(0, 1, 3, 2)))
        attn = self.softmax(attn_logits)
        attn = self.attn_dropout(attn)

        weighted_avg = kops.matmul(attn, v)
        weighted_avg = kops.transpose(weighted_avg, axes=(0, 2, 1, 3))
        weighted_avg = kops.reshape(weighted_avg, (batch_size, -1, self.num_heads * self.projection_dim))

        # Output projection
        output = self.proj(weighted_avg)
        output = self.linear_dropout(output)

        return output


class Transformer(Layer):
    def __init__(
        self,
        num_heads: int = 4,
        embedding_dim: int = 90,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        linear_drop: float = 0.0,
        attention_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.linear_drop = linear_drop
        self.attention_drop = attention_drop

        self.norm_1 = LayerNormalization(epsilon=1e-5)

        self.attn = MHSA(
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            qkv_bias=self.qkv_bias,
            attention_drop=self.attention_drop,
            linear_drop=dropout,
        )
        self.norm_2 = LayerNormalization(epsilon=1e-5)

        hidden_features = int(self.embedding_dim * self.mlp_ratio)

        self.mlp_block_0 = Dense(hidden_features, activation="swish")
        self.mlp_block_1 = Dropout(self.linear_drop)
        self.mlp_block_2 = Dense(embedding_dim)
        self.mlp_block_3 = Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        x = x + self.attn(self.norm_1(x))

        mlp_block_out = self.mlp_block_0(self.norm_2(x))
        mlp_block_out = self.mlp_block_1(mlp_block_out)
        mlp_block_out = self.mlp_block_2(mlp_block_out)
        mlp_block_out = self.mlp_block_3(mlp_block_out)

        x = x + mlp_block_out

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embedding_dim": self.embedding_dim,
                "qkv_bias": self.qkv_bias,
                "mlp_ratio": self.mlp_ratio,
                "dropout": self.dropout,
                "linear_drop": self.linear_drop,
                "attention_drop": self.attention_drop,
            }
        )
        return config