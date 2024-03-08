import torch.nn as nn
from modules.conv import *
from fns import normalize

class ConditionedUNet(nn.Module):
    def __init__(
        self, 
        n_channels: int,
        word_embed_size: int,
    ):
        super().__init__()
        self.n_channels = n_channels
        
        self.enc1 = ConditionedConvBlock(n_channels, 64, 11, word_embed_size)
        self.enc2 = ConditionedConvBlock(64, 128, 7, word_embed_size)
        self.enc3 = ConditionedConvBlock(128, 128, 5, word_embed_size)

        
        self.dec1 = TransposedConvBlock(128, 128, 5)
        self.dec2 = TransposedConvBlock(128*2, 64, 8)
        self.dec3 = TransposedConvBlock(64*2, 1, 11)
        
    def forward(self, imgs: torch.Tensor, words: torch.Tensor):
        x1 = self.enc1(imgs, words)
        x2 = self.enc2(x1, words)
        x3 = self.enc3(x2, words)
        h1 = self.dec1(x3)
        h2 = self.dec2(torch.concat([h1, x2], 1))
        h3 = self.dec3(torch.concat([h2, x1], 1))
        return h3
    
    
    
