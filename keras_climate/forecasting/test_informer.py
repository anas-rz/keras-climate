import numpy as np
import pytest
import keras

from keras_climate.forecasting.informer import Informer, DataEmbedding, DistillConv
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_informer_mapper


@pytest.mark.parametrize("distil", [True, False])
def test_builds_and_runs(distil):
    seq_len, label_len, pred_len, num_channels = 48, 24, 12, 5
    model = Informer(seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                      num_channels=num_channels, d_model=32, num_heads=4, d_ff=64,
                      encoder_layers=2, decoder_layers=1, dropout=0.0, distil=distil)
    enc_x = np.random.randn(2, seq_len, num_channels).astype("float32")
    dec_x = np.concatenate([
        enc_x[:, -label_len:, :],
        np.zeros((2, pred_len, num_channels), dtype="float32"),
    ], axis=1)
    y = keras.ops.convert_to_numpy(model([enc_x, dec_x]))
    assert y.shape == (2, pred_len, num_channels)


def test_distill_conv_halves_sequence_length():
    x = np.random.randn(2, 16, 8).astype("float32")
    out = DistillConv(8)(x)
    assert out.shape[1] == 8


def test_data_embedding_uses_matching_positional_slice():
    embed = DataEmbedding(d_model=8, max_len=100)
    x = np.random.randn(2, 12, 3).astype("float32")
    out = keras.ops.convert_to_numpy(embed(x))
    assert out.shape == (2, 12, 8)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    def sincos_pos_embed(max_len, dim):
        import math
        position = np.arange(max_len)[:, None]
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe = np.zeros((max_len, dim), dtype="float32")
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    class TorchValueEmbedding(nn.Module):
        def __init__(self, in_dim, d_model):
            super().__init__()
            self.conv = nn.Conv1d(in_dim, d_model, kernel_size=3, bias=False)

        def forward(self, x):
            left, right = x[:, -1:, :], x[:, :1, :]
            padded = torch.cat([left, x, right], dim=1).permute(0, 2, 1)
            return self.conv(padded).permute(0, 2, 1)

    class TorchDataEmbedding(nn.Module):
        def __init__(self, in_dim, d_model, max_len=200):
            super().__init__()
            self.value_embedding = TorchValueEmbedding(in_dim, d_model)
            self.register_buffer("position_embedding",
                                  torch.from_numpy(sincos_pos_embed(max_len, d_model)))

        def forward(self, x):
            L = x.shape[1]
            return self.value_embedding(x) + self.position_embedding[:L][None]

    class TorchDistillConv(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
            self.bn = nn.BatchNorm1d(d_model, eps=1e-5)
            self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.bn(x)
            x = F.elu(x)
            x = self.pool(x)
            return x.permute(0, 2, 1)

    class TorchMHA(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.scale = self.head_dim ** -0.5
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def split(self, x, B, L):
            return x.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def forward(self, query, key, value, mask=None):
            B, Lq, Lk = query.shape[0], query.shape[1], key.shape[1]
            q = self.split(self.q_proj(query), B, Lq)
            k = self.split(self.k_proj(key), B, Lk)
            v = self.split(self.v_proj(value), B, Lk)
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn = scores.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, Lq, -1)
            return self.out_proj(out)

    class TorchEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            self.attn = TorchMHA(d_model, num_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            x = self.norm1(x + self.attn(x, x, x))
            y = F.gelu(self.conv1(x.permute(0, 2, 1)))
            y = self.conv2(y).permute(0, 2, 1)
            return self.norm2(x + y)

    class TorchDecoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            self.self_attn = TorchMHA(d_model, num_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.cross_attn = TorchMHA(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.norm3 = nn.LayerNorm(d_model)

        def forward(self, x, enc_out, self_mask=None):
            x = self.norm1(x + self.self_attn(x, x, x, mask=self_mask))
            x = self.norm2(x + self.cross_attn(x, enc_out, enc_out))
            y = F.gelu(self.conv1(x.permute(0, 2, 1)))
            y = self.conv2(y).permute(0, 2, 1)
            return self.norm3(x + y)

    class TorchInformer(nn.Module):
        def __init__(self, seq_len, label_len, pred_len, num_channels, d_model, num_heads,
                     d_ff, encoder_layers, decoder_layers, distil):
            super().__init__()
            self.pred_len = pred_len
            self.distil = distil
            self.enc_embedding = TorchDataEmbedding(num_channels, d_model)
            self.dec_embedding = TorchDataEmbedding(num_channels, d_model)
            self.encoder_layers = nn.ModuleList(
                [TorchEncoderLayer(d_model, num_heads, d_ff) for _ in range(encoder_layers)])
            self.distills = nn.ModuleList(
                [TorchDistillConv(d_model) for _ in range(encoder_layers - 1)]) if distil else None
            self.decoder_layers = nn.ModuleList(
                [TorchDecoderLayer(d_model, num_heads, d_ff) for _ in range(decoder_layers)])
            self.projection = nn.Linear(d_model, num_channels)
            total_dec_len = label_len + pred_len
            mask = torch.triu(torch.ones(total_dec_len, total_dec_len) * -1e9, diagonal=1)
            self.register_buffer("self_mask", mask[None, None, :, :])

        def forward(self, enc_x, dec_x):
            enc_out = self.enc_embedding(enc_x)
            for i, layer in enumerate(self.encoder_layers):
                enc_out = layer(enc_out)
                if self.distil and i < len(self.encoder_layers) - 1:
                    enc_out = self.distills[i](enc_out)

            dec_out = self.dec_embedding(dec_x)
            for layer in self.decoder_layers:
                dec_out = layer(dec_out, enc_out, self_mask=self.self_mask)

            out = self.projection(dec_out)
            return out[:, -self.pred_len:, :]

    torch.manual_seed(0)
    seq_len, label_len, pred_len, num_channels = 32, 16, 8, 4
    d_model, num_heads, d_ff = 16, 2, 32
    encoder_layers, decoder_layers = 2, 1

    torch_model = TorchInformer(seq_len, label_len, pred_len, num_channels, d_model, num_heads,
                                 d_ff, encoder_layers, decoder_layers, distil=True)
    torch_model.eval()
    with torch.no_grad():
        for m in torch_model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.normal_(1.0, 0.1)
                m.bias.normal_(0.0, 0.1)
                m.running_mean.normal_(0.0, 0.1)
                m.running_var.uniform_(0.5, 1.5)
        for name, p in torch_model.named_parameters():
            if "bn" not in name:
                p.normal_(0.0, 0.2)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        if "num_batches_tracked" in k or k.endswith("position_embedding") or k.endswith("self_mask"):
            continue
        k = k.replace("encoder_layers.", "encoder_layer").replace("decoder_layers.", "decoder_layer")
        k = k.replace("distills.", "distill")
        state_dict[k] = v.detach().numpy()

    keras_model = Informer(seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                            num_channels=num_channels, d_model=d_model, num_heads=num_heads,
                            d_ff=d_ff, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                            dropout=0.0, distil=True)

    mapper = build_informer_mapper(encoder_layers=encoder_layers, decoder_layers=decoder_layers)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=False, verbose=False)
    assert set(report["missing_in_source"]) == {
        "enc_embedding/position_embedding", "dec_embedding/position_embedding",
    }
    assert not report["unused_source_keys"]

    enc_np = np.random.randn(2, seq_len, num_channels).astype("float32")
    dec_np = np.concatenate([
        enc_np[:, -label_len:, :], np.zeros((2, pred_len, num_channels), dtype="float32"),
    ], axis=1)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(enc_np), torch.from_numpy(dec_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model([enc_np, dec_np], training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"Informer weight port numerical mismatch: max abs diff {max_diff}"
