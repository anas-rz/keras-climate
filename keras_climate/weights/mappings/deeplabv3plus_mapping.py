import re


def _conv_bn_rules(torch_prefix, keras_prefix):
    tp = re.escape(torch_prefix)
    return [
        (rf"^{tp}\.conv\.weight$", f"{keras_prefix}/conv/kernel"),
        (rf"^{tp}\.bn\.weight$", f"{keras_prefix}/bn/gamma"),
        (rf"^{tp}\.bn\.bias$", f"{keras_prefix}/bn/beta"),
        (rf"^{tp}\.bn\.running_mean$", f"{keras_prefix}/bn/moving_mean"),
        (rf"^{tp}\.bn\.running_var$", f"{keras_prefix}/bn/moving_variance"),
    ]


def build_resnet_backbone_mapper(keras_prefix="backbone", layer_counts=(3, 4, 6, 3)):
    rules = [
        (r"^conv1\.weight$", f"{keras_prefix}_stem/conv/kernel"),
        (r"^bn1\.weight$", f"{keras_prefix}_stem/bn/gamma"),
        (r"^bn1\.bias$", f"{keras_prefix}_stem/bn/beta"),
        (r"^bn1\.running_mean$", f"{keras_prefix}_stem/bn/moving_mean"),
        (r"^bn1\.running_var$", f"{keras_prefix}_stem/bn/moving_variance"),
    ]

    for layer_idx in range(1, 5):
        num_blocks = layer_counts[layer_idx - 1]
        stage_kp = f"{keras_prefix}_stage{layer_idx}"
        for i in range(num_blocks):
            tp = f"layer{layer_idx}.{i}"
            kp = f"{stage_kp}_block{i}"
            for c in (1, 2):
                rules += [
                    (
                        rf"^{re.escape(tp)}\.conv{c}\.weight$",
                        f"{kp}_conv{c}/conv/kernel",
                    ),
                    (rf"^{re.escape(tp)}\.bn{c}\.weight$", f"{kp}_conv{c}/bn/gamma"),
                    (rf"^{re.escape(tp)}\.bn{c}\.bias$", f"{kp}_conv{c}/bn/beta"),
                    (
                        rf"^{re.escape(tp)}\.bn{c}\.running_mean$",
                        f"{kp}_conv{c}/bn/moving_mean",
                    ),
                    (
                        rf"^{re.escape(tp)}\.bn{c}\.running_var$",
                        f"{kp}_conv{c}/bn/moving_variance",
                    ),
                ]
            rules += [
                (rf"^{re.escape(tp)}\.conv3\.weight$", f"{kp}_conv3/kernel"),
                (rf"^{re.escape(tp)}\.bn3\.weight$", f"{kp}_bn3/gamma"),
                (rf"^{re.escape(tp)}\.bn3\.bias$", f"{kp}_bn3/beta"),
                (rf"^{re.escape(tp)}\.bn3\.running_mean$", f"{kp}_bn3/moving_mean"),
                (rf"^{re.escape(tp)}\.bn3\.running_var$", f"{kp}_bn3/moving_variance"),
                (
                    rf"^{re.escape(tp)}\.downsample\.0\.weight$",
                    f"{kp}_downsample_conv/kernel",
                ),
                (
                    rf"^{re.escape(tp)}\.downsample\.1\.weight$",
                    f"{kp}_downsample_bn/gamma",
                ),
                (
                    rf"^{re.escape(tp)}\.downsample\.1\.bias$",
                    f"{kp}_downsample_bn/beta",
                ),
                (
                    rf"^{re.escape(tp)}\.downsample\.1\.running_mean$",
                    f"{kp}_downsample_bn/moving_mean",
                ),
                (
                    rf"^{re.escape(tp)}\.downsample\.1\.running_var$",
                    f"{kp}_downsample_bn/moving_variance",
                ),
            ]

    return rules


def build_deeplabv3plus_mapper(
    backbone_keras_prefix="backbone", layer_counts=(3, 4, 6, 3)
):
    rules = build_resnet_backbone_mapper(backbone_keras_prefix, layer_counts)

    rules += _conv_bn_rules("aspp.b0", "aspp/b0")
    for r in (6, 12, 18):
        rules += _conv_bn_rules(f"aspp.b{r}", f"aspp/b{r}")
    rules += _conv_bn_rules("aspp.pool_conv", "aspp/pool_conv")
    rules += _conv_bn_rules("aspp.project", "aspp/project")

    rules += _conv_bn_rules("low_level_project", "low_level_project")
    rules += _conv_bn_rules("decoder_conv1", "decoder_conv1")
    rules += _conv_bn_rules("decoder_conv2", "decoder_conv2")

    rules += [
        (r"^logits\.weight$", "logits/kernel"),
        (r"^logits\.bias$", "logits/bias"),
    ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
