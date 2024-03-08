from torch import nn 
import torch 
import torch.nn.functional as F
from modules.ffn import FFN 



class ConditionedConvBlock(nn.Module):
    BATCH_SIZE = 50
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, word_embed_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.attn = nn.MultiheadAttention(in_channels, num_heads=1, batch_first=True, kdim=word_embed_size, vdim=word_embed_size)

    
    def forward(self, imgs: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        sizes = imgs.shape[-2:]
        feats = imgs.flatten(-2, -1).permute(0, 2, 1)
        feats = torch.concat([self.attn(feats[:, i:(i+self.BATCH_SIZE), :], words, words)[0] for i in range(0, feats.shape[1], self.BATCH_SIZE)], 1)
        feats = feats.view(feats.shape[0], *sizes, feats.shape[-1]).permute(0, 3, 1, 2)
        feats = self.conv(feats)
        return feats


class TransposedConvBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.up(imgs)


def normalize(x: torch.Tensor):
    minx = x.view(x.shape[0],-1).min(-1)[0].view(-1, 1, 1, 1)
    maxx = x.view(x.shape[0],-1).max(-1)[0].view(-1, 1, 1, 1)
    return (x-minx)/(maxx-minx)
    