import keras
from keras import layers, ops
from keras_climate.utils.layers import sincos_position_embedding


class ValueEmbedding(layers.Layer):

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.conv = layers.Conv1D(
            self.d_model, kernel_size=3, padding="valid", use_bias=False, name="conv"
        )
        super().build(input_shape)

    def call(self, x):
        left, right = x[:, -1:, :], x[:, :1, :]
        padded = ops.concatenate([left, x, right], axis=1)
        return self.conv(padded)


class DataEmbedding(layers.Layer):

    def __init__(self, d_model, max_len=5000, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_rate = dropout
        self._pos_np = sincos_position_embedding(max_len, d_model)

    def build(self, input_shape):
        self.value_embedding = ValueEmbedding(self.d_model, name="value_embedding")
        self.position_embedding = self.add_weight(
            shape=self._pos_np.shape,
            initializer=keras.initializers.Constant(self._pos_np),
            trainable=False,
            name="position_embedding",
        )
        self.drop = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        L = x.shape[1]
        out = self.value_embedding(x) + self.position_embedding[None, :L, :]
        return self.drop(out, training=training)


class DistillConv(layers.Layer):

    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.pad1 = layers.ZeroPadding1D(1, name="pad1")
        self.conv = layers.Conv1D(
            self.d_model,
            kernel_size=3,
            strides=1,
            padding="valid",
            use_bias=False,
            name="conv",
        )
        self.bn = layers.BatchNormalization(epsilon=1e-5, name="bn")
        self.act = layers.Activation("elu")
        self.pad2 = layers.ZeroPadding1D(1, name="pad2")
        self.pool = layers.MaxPooling1D(
            pool_size=3, strides=2, padding="valid", name="pool"
        )
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.pad1(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        x = self.pad2(x)
        return self.pool(x)


class MultiHeadAttention(layers.Layer):

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.d_model, name="q_proj")
        self.k_proj = layers.Dense(self.d_model, name="k_proj")
        self.v_proj = layers.Dense(self.d_model, name="v_proj")
        self.out_proj = layers.Dense(self.d_model, name="out_proj")
        super().build(input_shape)

    def _split_heads(self, x, B, L):
        x = ops.reshape(x, (B, L, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, query, key, value, mask=None):
        B = ops.shape(query)[0]
        Lq, Lk = ops.shape(query)[1], ops.shape(key)[1]
        q = self._split_heads(self.q_proj(query), B, Lq)
        k = self._split_heads(self.k_proj(key), B, Lk)
        v = self._split_heads(self.v_proj(value), B, Lk)

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, Lq, self.d_model))
        return self.out_proj(out)


class InformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(d_model, num_heads, name="attn")
        self.norm1 = layers.LayerNormalization(name="norm1")
        self.conv1 = layers.Conv1D(d_ff, 1, name="conv1")
        self.act = layers.Activation("gelu")
        self.conv2 = layers.Conv1D(d_model, 1, name="conv2")
        self.norm2 = layers.LayerNormalization(name="norm2")
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out, training=training))
        y = self.drop(self.act(self.conv1(x)), training=training)
        y = self.drop(self.conv2(y), training=training)
        return self.norm2(x + y)


class InformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = MultiHeadAttention(d_model, num_heads, name="self_attn")
        self.norm1 = layers.LayerNormalization(name="norm1")
        self.cross_attn = MultiHeadAttention(d_model, num_heads, name="cross_attn")
        self.norm2 = layers.LayerNormalization(name="norm2")
        self.conv1 = layers.Conv1D(d_ff, 1, name="conv1")
        self.act = layers.Activation("gelu")
        self.conv2 = layers.Conv1D(d_model, 1, name="conv2")
        self.norm3 = layers.LayerNormalization(name="norm3")
        self.drop = layers.Dropout(dropout)

    def call(self, x, enc_out, self_mask=None, training=False):
        x = self.norm1(
            x + self.drop(self.self_attn(x, x, x, mask=self_mask), training=training)
        )
        x = self.norm2(
            x + self.drop(self.cross_attn(x, enc_out, enc_out), training=training)
        )
        y = self.drop(self.act(self.conv1(x)), training=training)
        y = self.drop(self.conv2(y), training=training)
        return self.norm3(x + y)


def causal_mask(seq_len):
    mask = ops.triu(ops.ones((seq_len, seq_len)) * -1e9, k=1)
    return mask[None, None, :, :]


def Informer(
    seq_len=96,
    label_len=48,
    pred_len=24,
    num_channels=7,
    d_model=64,
    num_heads=4,
    d_ff=128,
    encoder_layers=2,
    decoder_layers=1,
    dropout=0.1,
    distil=True,
    name="informer",
):
    enc_in = keras.Input((seq_len, num_channels), name="encoder_series")
    dec_in = keras.Input((label_len + pred_len, num_channels), name="decoder_series")

    enc_x = DataEmbedding(d_model, dropout=dropout, name="enc_embedding")(enc_in)
    for i in range(encoder_layers):
        enc_x = InformerEncoderLayer(
            d_model, num_heads, d_ff, dropout, name=f"encoder_layer{i}"
        )(enc_x)
        if distil and i < encoder_layers - 1:
            enc_x = DistillConv(d_model, name=f"distill{i}")(enc_x)

    dec_x = DataEmbedding(d_model, dropout=dropout, name="dec_embedding")(dec_in)
    self_mask = causal_mask(label_len + pred_len)
    for i in range(decoder_layers):
        dec_x = InformerDecoderLayer(
            d_model, num_heads, d_ff, dropout, name=f"decoder_layer{i}"
        )(dec_x, enc_x, self_mask=self_mask)

    out = layers.Dense(num_channels, name="projection")(dec_x)
    out = layers.Lambda(lambda t: t[:, -pred_len:, :], name="slice_forecast")(out)

    return keras.Model([enc_in, dec_in], out, name=name)
