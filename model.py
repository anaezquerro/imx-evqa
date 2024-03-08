import torch 
from torch import nn 
from transformers import AutoModel
from typing import Tuple, Optional
from modules import FFN, LSTM, ConditionedUNet, Encoder
from fns import to 

class VQAModel(nn.Module):

    def __init__(
        self, 
        img_size: Tuple[int, int],
        word_embed_size: int, 
        max_words: int, 
        pad_index: int, 
        n_answers: int,
        weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.img_size = img_size
        
        self.word_embed = nn.Embedding(max_words, word_embed_size, pad_index)
        self.question = LSTM(word_embed_size, word_embed_size, num_layers=1, bidirectional=True)
        self.img_weights = ConditionedUNet(3, word_embed_size)
        self.img_embed = Encoder(img_size, 3, word_embed_size)
        self.ffn = FFN(torch.prod(self.img_embed.hidden_dims).item() + word_embed_size, n_answers)
        self.mask_criteria = nn.BCEWithLogitsLoss()
        self.criteria = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, imgs: torch.Tensor, words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        word_embed = self.word_embed(words)
        word_embed, question = self.question(word_embed)
        weights = self.img_weights(imgs, word_embed)
        img_embed = self.img_embed(imgs*(weights > 0), word_embed).flatten(1, 3)
        feats = torch.cat([img_embed, question], -1)
        return self.ffn(feats), weights.unsqueeze(1)
    
    def loss(self, s_answer: torch.Tensor, s_mask: torch.Tensor, answers: torch.Tensor, masks: torch.Tensor):
        return self.criteria(s_answer, answers) + self.mask_criteria(s_mask.flatten(), masks.flatten().to(torch.float32))
        
    
    def feedback(self, imgs: torch.Tensor, words: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        word_embed = self.word_embed(words)
        word_embed, question = self.question(word_embed)
        img_embed = self.img_embed(imgs*mask, word_embed).flatten(1, 3)
        feats = torch.cat([img_embed, question], -1)
        return self.ffn(feats)

        
        
        
        