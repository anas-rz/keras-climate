from keras_climate.weights.mappings.vit_mapping import build_vit_mapper


def build_scalemae_mapper(keras_prefix="scalemae_encoder"):
    return build_vit_mapper(keras_prefix)


SCALEMAE_SKIP_PATTERNS = [r"^pos_embed$", r"^decoder_pos_embed$"]
