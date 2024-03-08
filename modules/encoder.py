from torch import nn 
from modules.conv import * 
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, img_sizes: Tuple[int, int], n_channels: int, word_embed_size: int, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.down1 = ConditionedConvBlock(n_channels, 64, 11, word_embed_size)
        self.down2 = ConditionedConvBlock(64, 128, 7, word_embed_size)
        self.down3 = ConditionedConvBlock(128, 256, 5, word_embed_size)
        
        words = torch.empty(1, 10, word_embed_size)
        imgs = torch.empty(1, n_channels, *img_sizes)
        out = self.forward(imgs, words)
        self.hidden_dims = torch.tensor(out.shape[1:])

    def forward(self, imgs: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(imgs, words)
        x2 = self.down2(x1, words)
        x3 = self.down3(x2, words)
        return x3