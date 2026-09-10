# model/mini_vla.py

import torch
import torch.nn as nn
from model.vision_encoder import VisionEncoder
from model.transformer import TransformerBlock
from model.rmsnorm import RMSNorm

class MiniVLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_encoder = VisionEncoder(
            image_size = config.image_size,
            image_dim = config.image_dim,
            patch_size = config.patch_size
        )

        if config.image_dim != config.n_embed:
            self.image_proj = nn.Linear(config.image_dim, config.n_embed)
        else:
            self.image_proj = nn.Identity()

        # text: [B, text_len]
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.max_text_len = self.config.max_text_len

        # use the action to output the action, it store the information of image and text
        # shape [batch, seq_len, embed_dim] = [1, 1, embed_dim]
        self.action_token = nn.Parameter(torch.zeros(1, 1, config.n_embed))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = RMSNorm(config.n_embed)
        self.action_head = nn.Linear(
            config.n_embed,
            config.action_dim
        )

    def forward(self, image, text, mask = None):
        # image shape: [batch, channels, height, width]
        # text shape: [batch, text_len]
        # mask shape: [batch, text_len] or None

        image_tokens = self.image_proj(self.vision_encoder(image))

        text_tokens = self.token_embedding(text)

        # expand the action token to the batch size
        action_token = self.action_token.expand(image_tokens.size(0), -1, -1)

        # shape x: [batch, image_seq_len + text_seq_len + 1, embed_dim]
        x = torch.cat([image_tokens, text_tokens, action_token], dim = 1)

        image_mask = torch.ones(
            image_tokens.size(0),
            image_tokens.size(1),
            device = x.device
        )

        attention_mask = torch.ones(
            image_tokens.size(0),
            1,
            device = x.device
        )

        # shape mask: [batch, image_seq_len + text_seq_len + 1]
        full_mask = torch.cat([image_mask, mask, attention_mask], dim = 1)

        for block in self.transformer_blocks:
            x, _ = block(
                x,
                use_cache = False,
                past_kv = None,
                use_causal_mask = False,
                attention_mask = full_mask
            )

        # x shape: [batch, seq_len, embed_dim]
        # just do the mlp for the last token
        return self.action_head(self.norm(x[:,-1,:]))
