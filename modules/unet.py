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
        
        self.enc1 = ConditionedConvBlock(n_channels, 16, 3, word_embed_size)
        self.enc2 = ConditionedConvBlock(16, 32, 3, word_embed_size)
        
        self.dec1 = TransposedConvBlock(32, 16, 4)
        self.dec2 = TransposedConvBlock(16, 1, 3)
        
    def forward(self, imgs: torch.Tensor, words: torch.Tensor):
        x = self.enc1(imgs, words)
        h = self.enc2(x, words)
        h1 = self.dec1(h)
        h2 = self.dec2(h1)
        return normalize(h2)
    
    
    
