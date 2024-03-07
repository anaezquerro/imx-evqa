import torch 
from torch import nn 
from transformers import AutoModel
from typing import Tuple
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
    ):
        super().__init__()
        self.word_embed_size = word_embed_size
        self.img_size = img_size
        
        self.word_embed = nn.Embedding(max_words, word_embed_size, pad_index)
        self.question = LSTM(word_embed_size, word_embed_size, num_layers=1, bidirectional=False)
        self.img_weights = ConditionedUNet(3, word_embed_size)
        self.img_embed = Encoder(img_size, 3, word_embed_size)
        self.ffn = FFN(torch.prod(self.img_embed.hidden_dims).item() + word_embed_size, n_answers)
        self.criteria = nn.CrossEntropyLoss()
        
    def forward(self, imgs: torch.Tensor, words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        word_embed = self.word_embed(words)
        question = self.question(word_embed).squeeze(0)
        weights = (self.img_weights(imgs, word_embed) > 0.5)*1.0
        img_embed = self.img_embed(imgs*weights, word_embed).flatten(1, 3)
        feats = torch.cat([img_embed, question], -1)
        return self.ffn(feats), weights
    
    def loss(self, s_answer: torch.Tensor, answers: torch.Tensor):
        return self.criteria(s_answer, answers)
        
    
        
        
        
        