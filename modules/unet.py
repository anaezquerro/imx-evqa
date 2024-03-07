import torch.nn as nn
from modules.conv import *
from typing import Tuple

class ConditionedUNet(nn.Module):
    def __init__(
        self, 
        n_channels: int,
        word_embed_size: int,
        bilinear=False
    ):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        # encoder part 
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = ConditionedDown(64, 128, word_embed_size)
        self.down4 = ConditionedDown(128, 256 // factor, word_embed_size)
        
        # decoder        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16)
        
    def forward(self, imgs: torch.Tensor, words: torch.Tensor):
        x1 = self.inc(imgs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3, words)
        x5 = self.down4(x4, words)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
