# model/vision_encoder.py
import torch
from torch import nn

class VisionEncoder(nn.Module):
    def __init__(self, image_size = 224, image_dim = 256, patch_size = 16):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.image_size = image_size
        self.patch_size = patch_size
        self.image_dim = image_dim

        # the cnn kernel size is 16x16
        # stride is 16
        # total patches = (224 / 16) ^ 2 = 196
        # token length = 196
        self.num_patches = (image_size // patch_size) ** 2

        # input [chennels, height, width] = [3, 224, 224]
        # con2d   (channels, out_channels, kernel_size, stride)
        self.patch_embed = nn.Conv2d(
            in_channels = 3,
            out_channels = image_dim,
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self, x):
        # input x: [batch, channels, height, width]
        # output x: [batch, num_patches, image_dim]

        # [batch, channels, height, width] -> [batch, image_dim, height // patch_size, width // patch_size]
        x = self.patch_embed(x)

        # flatten the height and width dimensions
        # [batch, image_dim, height // patch_size, width // patch_size] -> [batch, image_dim, num_patches]
        # flatten from the dim 2 to the last dim
        x = x.flatten(2)

        # transpose the image_dim and num_patches dimensions
        # put the dim at the last dimension
        # [batch, image_dim, num_patches] -> [batch, num_patches, image_dim]
        x = x.transpose(1, 2)

        return x


if __name__ == "__main__":
    # test the vision encoder
    image_size = 224
    image_dim = 256
    patch_size = 16

    vision_encoder = VisionEncoder(image_size, image_dim, patch_size)

    # create a random image tensor
    # [batch, channels, height, width] = [1, 3, 224, 224]
    x = torch.randn(1, 3, image_size, image_size)

    # forward pass
    out = vision_encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
