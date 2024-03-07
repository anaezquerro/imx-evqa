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
        devices: Tuple[str, str]
    ):
        super().__init__()
        self.word_embed = nn.Embedding(max_words, word_embed_size, pad_index).to(devices[0])
        self.img_weights = ConditionedUNet(3, word_embed_size).to(devices[0])
        self.img_embed = Encoder(img_size, 3, word_embed_size).to(devices[1])
        self.ffn = FFN(torch.prod(self.img_embed.hidden_dims).item(), n_answers, activation=nn.Softmax(-1)).to(devices[1])
        self.criteria = nn.CrossEntropyLoss()
        self.devices = devices
        
    def forward(self, imgs: torch.Tensor, words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        word_embed = self.word_embed(words)
        weights = self.img_weights(imgs, word_embed)
        imgs, weights, word_embed = to(self.devices[1], imgs, weights, word_embed)
        img_embed = self.img_embed(imgs*weights, word_embed).flatten(1, 3)
        return self.ffn(img_embed), weights.mean()
    
    def loss(self, s_answer: torch.Tensor, answers: torch.Tensor):
        return self.criteria(s_answer, answers.to(self.devices[1]))
        
    
        
        
        
        