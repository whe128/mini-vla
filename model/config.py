from dataclasses import dataclass

@dataclass
class VLAConfig:
    """ base GPT config, params common to all GPT versions """

    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embed: int = 256
    dropout: int = 0.0

    image_size: int = 224
    image_dim: int = 256
    patch_size: int = 16

    action_dim: int = 7
    max_text_len: int = 20
