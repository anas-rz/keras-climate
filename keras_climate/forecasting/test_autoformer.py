import numpy as np
import pytest
import keras

from keras_climate.forecasting.autoformer import Autoformer, AutoCorrelation
from keras_climate.weights import WeightConverter
from keras_climate.weights.mappings import build_autoformer_mapper


def test_builds_and_runs():
    seq_len, label_len, pred_len, num_channels = 48, 24, 12, 5
    model = Autoformer(seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                        num_channels=num_channels, d_model=32, num_heads=4, d_ff=64,
                        encoder_layers=2, decoder_layers=1, moving_avg=5, dropout=0.0)
    enc_x = np.random.randn(2, seq_len, num_channels).astype("float32")
    label_x = enc_x[:, -label_len:, :]
    y = keras.ops.convert_to_numpy(model([enc_x, label_x]))
    assert y.shape == (2, pred_len, num_channels)


def test_auto_correlation_same_length_self_attention():
    layer = AutoCorrelation(d_model=8, num_heads=2)
    x = np.random.randn(2, 10, 8).astype("float32")
    out = keras.ops.convert_to_numpy(layer(x, x, x))
    assert out.shape == (2, 10, 8)


def test_auto_correlation_cross_attention_different_lengths():
    layer = AutoCorrelation(d_model=8, num_heads=2)
    query = np.random.randn(2, 12, 8).astype("float32")
    key = np.random.randn(2, 20, 8).astype("float32")
    out = keras.ops.convert_to_numpy(layer(query, key, key))
    assert out.shape == (2, 12, 8)


def test_weight_port_roundtrip_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    def rfft_np_style(x):
        X = torch.fft.rfft(x, dim=-1)
        return X.real, X.imag

    def irfft_np_style(re, im, n):
        X = torch.complex(re, im)
        return torch.fft.irfft(X, n=n, dim=-1)

    class TorchValueEmbedding(nn.Module):
        def __init__(self, in_dim, d_model):
            super().__init__()
            self.conv = nn.Conv1d(in_dim, d_model, kernel_size=3, bias=False)

        def forward(self, x):
            left, right = x[:, -1:, :], x[:, :1, :]
            padded = torch.cat([left, x, right], dim=1).permute(0, 2, 1)
            return self.conv(padded).permute(0, 2, 1)

    class TorchSeriesDecomp(nn.Module):
        def __init__(self, kernel_size):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

        def forward(self, x):
            pad_front = (self.kernel_size - 1) // 2
            pad_end = self.kernel_size - 1 - pad_front
            front = x[:, :1, :].repeat(1, pad_front, 1)
            end = x[:, -1:, :].repeat(1, pad_end, 1)
            padded = torch.cat([front, x, end], dim=1)
            trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)
            return x - trend, trend

    class TorchAutoCorrelation(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def split(self, x, B, L):
            return x.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        def forward(self, query, key, value):
            B, Lq, Lk = query.shape[0], query.shape[1], key.shape[1]
            q = self.split(self.q_proj(query), B, Lq)
            k = self.split(self.k_proj(key), B, Lk)
            v = self.split(self.v_proj(value), B, Lk)

            if Lk > Lq:
                k, v = k[:, :, :Lq, :], v[:, :, :Lq, :]
            elif Lk < Lq:
                pad = Lq - Lk
                zeros = torch.zeros(B, self.num_heads, pad, self.head_dim)
                k = torch.cat([k, zeros], dim=2)
                v = torch.cat([v, zeros], dim=2)

            q_t, k_t = q.permute(0, 1, 3, 2), k.permute(0, 1, 3, 2)
            q_re, q_im = rfft_np_style(q_t)
            k_re, k_im = rfft_np_style(k_t)
            corr_re = q_re * k_re + q_im * k_im
            corr_im = q_im * k_re - q_re * k_im
            corr = irfft_np_style(corr_re, corr_im, Lq)

            weights = torch.softmax(corr.mean(dim=2), dim=-1)

            v_t = v.permute(0, 1, 3, 2)
            w_re, w_im = rfft_np_style(weights)
            v_re, v_im = rfft_np_style(v_t)
            w_re, w_im = w_re.unsqueeze(2), w_im.unsqueeze(2)
            out_re = v_re * w_re - v_im * w_im
            out_im = v_re * w_im + v_im * w_re
            agg = irfft_np_style(out_re, out_im, Lq)
            agg = agg.permute(0, 1, 3, 2)

            out = agg.permute(0, 2, 1, 3).reshape(B, Lq, -1)
            return self.out_proj(out)

    class TorchEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, moving_avg):
            super().__init__()
            self.auto_correlation = TorchAutoCorrelation(d_model, num_heads)
            self.decomp1 = TorchSeriesDecomp(moving_avg)
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.decomp2 = TorchSeriesDecomp(moving_avg)

        def forward(self, x):
            x = x + self.auto_correlation(x, x, x)
            x, _ = self.decomp1(x)
            y = F.gelu(self.conv1(x.permute(0, 2, 1)))
            y = self.conv2(y).permute(0, 2, 1)
            x, _ = self.decomp2(x + y)
            return x

    class TorchDecoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, num_channels, moving_avg):
            super().__init__()
            self.self_correlation = TorchAutoCorrelation(d_model, num_heads)
            self.decomp1 = TorchSeriesDecomp(moving_avg)
            self.cross_correlation = TorchAutoCorrelation(d_model, num_heads)
            self.decomp2 = TorchSeriesDecomp(moving_avg)
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.decomp3 = TorchSeriesDecomp(moving_avg)
            self.trend_proj = nn.Linear(d_model, num_channels, bias=False)

        def forward(self, x, cross):
            x = x + self.self_correlation(x, x, x)
            x, trend1 = self.decomp1(x)
            x = x + self.cross_correlation(x, cross, cross)
            x, trend2 = self.decomp2(x)
            y = F.gelu(self.conv1(x.permute(0, 2, 1)))
            y = self.conv2(y).permute(0, 2, 1)
            x, trend3 = self.decomp3(x + y)
            trend = self.trend_proj(trend1 + trend2 + trend3)
            return x, trend

    class TorchAutoformer(nn.Module):
        def __init__(self, seq_len, label_len, pred_len, num_channels, d_model, num_heads,
                     d_ff, encoder_layers, decoder_layers, moving_avg):
            super().__init__()
            self.pred_len = pred_len
            self.label_len = label_len
            self.num_channels = num_channels
            self.init_decomp = TorchSeriesDecomp(moving_avg)
            self.enc_embedding = TorchValueEmbedding(num_channels, d_model)
            self.dec_embedding = TorchValueEmbedding(num_channels, d_model)
            self.encoder_layers = nn.ModuleList(
                [TorchEncoderLayer(d_model, num_heads, d_ff, moving_avg) for _ in range(encoder_layers)])
            self.decoder_layers = nn.ModuleList(
                [TorchDecoderLayer(d_model, num_heads, d_ff, num_channels, moving_avg)
                 for _ in range(decoder_layers)])
            self.projection = nn.Linear(d_model, num_channels)

        def forward(self, enc_x, label_x):
            seasonal_init, trend_init = self.init_decomp(label_x)
            mean_trend = label_x.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
            trend_init_full = torch.cat([trend_init, mean_trend], dim=1)
            zeros_seasonal = torch.zeros(label_x.shape[0], self.pred_len, self.num_channels)
            seasonal_init_full = torch.cat([seasonal_init, zeros_seasonal], dim=1)

            enc_out = self.enc_embedding(enc_x)
            for layer in self.encoder_layers:
                enc_out = layer(enc_out)

            dec_out = self.dec_embedding(seasonal_init_full)
            trend = trend_init_full
            for layer in self.decoder_layers:
                dec_out, layer_trend = layer(dec_out, enc_out)
                trend = trend + layer_trend

            seasonal_out = self.projection(dec_out)
            out = seasonal_out + trend
            return out[:, -self.pred_len:, :]

    torch.manual_seed(0)
    seq_len, label_len, pred_len, num_channels = 32, 16, 8, 4
    d_model, num_heads, d_ff, moving_avg = 16, 2, 32, 5
    encoder_layers, decoder_layers = 2, 1

    torch_model = TorchAutoformer(seq_len, label_len, pred_len, num_channels, d_model, num_heads,
                                   d_ff, encoder_layers, decoder_layers, moving_avg)
    torch_model.eval()
    with torch.no_grad():
        for p in torch_model.parameters():
            p.normal_(0.0, 0.2)

    state_dict = {}
    for k, v in torch_model.state_dict().items():
        k = k.replace("encoder_layers.", "encoder_layer").replace("decoder_layers.", "decoder_layer")
        state_dict[k] = v.detach().numpy()

    keras_model = Autoformer(seq_len=seq_len, label_len=label_len, pred_len=pred_len,
                              num_channels=num_channels, d_model=d_model, num_heads=num_heads,
                              d_ff=d_ff, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
                              moving_avg=moving_avg, dropout=0.0)

    mapper = build_autoformer_mapper(encoder_layers=encoder_layers, decoder_layers=decoder_layers)
    report = WeightConverter(keras_model, state_dict, mapper).convert(strict=True, verbose=False)
    assert not report["missing_in_source"]
    assert not report["unused_source_keys"]

    enc_np = np.random.randn(2, seq_len, num_channels).astype("float32")
    label_np = enc_np[:, -label_len:, :]

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(enc_np), torch.from_numpy(label_np)).numpy()

    keras_out = keras.ops.convert_to_numpy(keras_model([enc_np, label_np], training=False))

    max_diff = np.abs(torch_out - keras_out).max()
    assert max_diff < 1e-2, f"Autoformer weight port numerical mismatch: max abs diff {max_diff}"
