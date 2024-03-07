from torch import nn 
from modules.conv import * 
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, img_sizes: Tuple[int, int], n_channels: int, word_embed_size: int, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = ConditionedDown(256, 512, word_embed_size)
        self.down4 = ConditionedDown(512, 1, word_embed_size)
        
        words = torch.empty(1, 10, word_embed_size)
        imgs = torch.empty(1, n_channels, *img_sizes)
        out = self.forward(imgs, words)
        self.hidden_dims = torch.tensor(out.shape[1:])

    def forward(self, imgs: torch.Tensor, words: torch.Tensor):
        x1 = self.inc(imgs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3, words)
        x5 = self.down4(x4, words)
        return x5