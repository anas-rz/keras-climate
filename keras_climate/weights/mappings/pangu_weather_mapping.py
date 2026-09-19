import re


def convert_pangu_weather_state_dict(flat_state_dict):
    out = {}
    for k, v in flat_state_dict.items():
        if (
            k.endswith((".conv.weight", ".conv_surface.weight"))
            and v.ndim == 3
            and v.shape[-1] == 1
        ):
            v = v[:, :, 0]
        out[k] = v
    return out


def _block_rules(i, j):
    prefix = f"layers.EarthSpecificLayer{i}.blocks.EarthSpecificBlock{j}"
    kp = f"layers_EarthSpecificLayer{i}/blocks_EarthSpecificBlock{j}"
    rules = [
        (rf"^{prefix}\.norm1\.weight$", f"{kp}/norm1/gamma"),
        (rf"^{prefix}\.norm1\.bias$", f"{kp}/norm1/beta"),
        (rf"^{prefix}\.norm2\.weight$", f"{kp}/norm2/gamma"),
        (rf"^{prefix}\.norm2\.bias$", f"{kp}/norm2/beta"),
        (rf"^{prefix}\.linear\.linear1\.weight$", f"{kp}/linear/linear1/kernel"),
        (rf"^{prefix}\.linear\.linear1\.bias$", f"{kp}/linear/linear1/bias"),
        (rf"^{prefix}\.linear\.linear2\.weight$", f"{kp}/linear/linear2/kernel"),
        (rf"^{prefix}\.linear\.linear2\.bias$", f"{kp}/linear/linear2/bias"),
        (
            rf"^{prefix}\.attention\.earth_specific_bias$",
            f"{kp}/attention/earth_specific_bias",
        ),
        (rf"^{prefix}\.attention\.linear1\.weight$", f"{kp}/attention/linear1/kernel"),
        (rf"^{prefix}\.attention\.linear1\.bias$", f"{kp}/attention/linear1/bias"),
        (rf"^{prefix}\.attention\.linear2\.weight$", f"{kp}/attention/linear2/kernel"),
        (rf"^{prefix}\.attention\.linear2\.bias$", f"{kp}/attention/linear2/bias"),
    ]
    return rules


def build_pangu_weather_mapper(depths=(2, 6, 6, 2)):
    rules = [
        (r"^_input_layer\.conv\.weight$", "_input_layer/conv/kernel"),
        (r"^_input_layer\.conv\.bias$", "_input_layer/conv/bias"),
        (r"^_input_layer\.conv_surface\.weight$", "_input_layer/conv_surface/kernel"),
        (r"^_input_layer\.conv_surface\.bias$", "_input_layer/conv_surface/bias"),
        (r"^downsample\.linear\.weight$", "downsample/linear/kernel"),
        (r"^downsample\.norm\.weight$", "downsample/norm/gamma"),
        (r"^downsample\.norm\.bias$", "downsample/norm/beta"),
        (r"^upsample\.linear1\.weight$", "upsample/linear1/kernel"),
        (r"^upsample\.linear2\.weight$", "upsample/linear2/kernel"),
        (r"^upsample\.norm\.weight$", "upsample/norm/gamma"),
        (r"^upsample\.norm\.bias$", "upsample/norm/beta"),
        (r"^_output_layer\.conv\.weight$", "_output_layer/conv/kernel"),
        (r"^_output_layer\.conv\.bias$", "_output_layer/conv/bias"),
        (r"^_output_layer\.conv_surface\.weight$", "_output_layer/conv_surface/kernel"),
        (r"^_output_layer\.conv_surface\.bias$", "_output_layer/conv_surface/bias"),
    ]
    for i, depth in enumerate(depths):
        for j in range(depth):
            rules += _block_rules(i, j)

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
