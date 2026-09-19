import re


def build_afno_mapper(depth, keras_prefix=""):
    kp = f"{keras_prefix}/" if keras_prefix else ""
    rules = [
        (r"^patch_embed\.proj\.weight$", f"{kp}patch_embed_proj/kernel"),
        (r"^patch_embed\.proj\.bias$", f"{kp}patch_embed_proj/bias"),
        (r"^head\.weight$", f"{kp}head/kernel"),
        (r"^head\.bias$", f"{kp}head/bias"),
    ]
    for i in range(depth):
        prefix = f"blocks.{i}"
        keras_block = f"{kp}block{i}"
        rules += [
            (rf"^{prefix}\.norm1\.weight$", f"{keras_block}/norm1/gamma"),
            (rf"^{prefix}\.norm1\.bias$", f"{keras_block}/norm1/beta"),
            (rf"^{prefix}\.filter\.w1$", f"{keras_block}/filter/w1"),
            (rf"^{prefix}\.filter\.b1$", f"{keras_block}/filter/b1"),
            (rf"^{prefix}\.filter\.w2$", f"{keras_block}/filter/w2"),
            (rf"^{prefix}\.filter\.b2$", f"{keras_block}/filter/b2"),
            (rf"^{prefix}\.norm2\.weight$", f"{keras_block}/norm2/gamma"),
            (rf"^{prefix}\.norm2\.bias$", f"{keras_block}/norm2/beta"),
            (rf"^{prefix}\.mlp\.fc1\.weight$", f"{keras_block}/mlp/fc1/kernel"),
            (rf"^{prefix}\.mlp\.fc1\.bias$", f"{keras_block}/mlp/fc1/bias"),
            (rf"^{prefix}\.mlp\.fc2\.weight$", f"{keras_block}/mlp/fc2/kernel"),
            (rf"^{prefix}\.mlp\.fc2\.bias$", f"{keras_block}/mlp/fc2/bias"),
        ]

    compiled = [(re.compile(p), r) for p, r in rules]

    def mapper(torch_key):
        for pattern, repl in compiled:
            if pattern.match(torch_key):
                return pattern.sub(repl, torch_key)
        return None

    return mapper
