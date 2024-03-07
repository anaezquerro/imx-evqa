
from torch import nn 
from typing import Optional
from transformers import AutoModel
from modules.ffn import FFN 
import torch 
from torch.nn.utils.rnn import pad_sequence

class PretrainedEmbedding(nn.Module):
    def __init__(
        self, 
        pretrained: str,
        pad_index: int,
        embed_size: Optional[int] = None,
        finetune: bool = True, 
        dropout: float = 0,
        **_
    ):
        super().__init__()
        self.embed = AutoModel.from_pretrained(pretrained).requires_grad_(finetune)
        self.embed_size = embed_size or self.embed.config.hidden_size
        self.proj = nn.Sequential(
            FFN(self.embed.config.hidden_size, embed_size) if embed_size else nn.Identity(),
            nn.Dropout(dropout)
        )
        self.pad_index = pad_index
        
    def forward(self, words: torch.Tensor) -> torch.Tensor:
        """Forward pass of the pretrained embedding.

        Args:
            words (torch.Tensor): ``[batch_size, seq_len, fix_len]``.
            
        Returns:
            embed (torch.Tensor): ``[batch_size, seq_len, embed_size]``.
        """
        mask = words != self.pad_index
        lens = mask.sum((-2, -1)).tolist()
        flat = pad_sequence(words[mask].split(lens), True, self.pad_index)
        flat_mask = flat != self.pad_index
        embed = self.embed(flat, attention_mask=flat_mask).last_hidden_state
        embed = list(embed[flat_mask].split(flat_mask.sum(-1).tolist()))
        
        for i in range(len(embed)):
            m = mask[i].sum(-1)
            m = m[m > 0].tolist()
            embed[i] = torch.stack([x.mean(0) for x in embed[i].split(m)], 0)
        
        embed = pad_sequence(embed, True, 0)
        return self.proj(embed)
