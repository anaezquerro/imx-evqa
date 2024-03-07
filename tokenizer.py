
from typing import Optional, List
import torch, pickle
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:
    EXTENSION = 'tkz'
    TRAINABLE = True 
    PARAMS = ['field', 'pad_token', 'unk_token', 'bos_token', 'eos_token', 'lower', 'max_words', 'counter']
    
    def __init__(
        self, 
        field: str,
        pad_token: Optional[str] = '<pad>',
        unk_token: Optional[str] = '<unk>',
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        lower: bool = False,
        max_words: Optional[int] = None
    ):
        self.field = field 
        self.pad_token = pad_token 
        self.unk_token = unk_token 
        self.bos_token = bos_token  
        self.eos_token = eos_token 
        self.lower = lower
        self.max_words = max_words
        
        self.specials = [token for token in (pad_token, unk_token, self.bos_token, self.eos_token) if isinstance(token, str)]
        self.counter = dict()
        self.reset()
        
    def save(self, path: str):
        data = {param: getattr(self, param) for param in self.PARAMS}
        with open(path, 'wb') as writer:
            pickle.dump(data, writer)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as reader:
            data = pickle.load(reader)
        counter = data.pop('counter')
        tkz = cls(**data)
        tkz.counter = counter 
        tkz.load_counter()
        return tkz 
        
        
    def reset(self):
        self.vocab = {token: i for i, token in enumerate(self.specials)}
        self.inv_vocab = {index: token for token, index in self.vocab.items()}
        
    def __len__(self) -> int:
        return len(self.vocab)
    
    def preprocess(self, token: str) -> str:
        return token.lower() if self.lower else token 
    
    def add(self, token: str):
        try:
            return self.vocab[token]
        except KeyError:
            self.vocab[token] = len(self.vocab)
            self.inv_vocab[self.vocab[token]] = token
            return self.vocab[token]
        
    def count(self, token: str):
        try:
            self.counter[token] +=1 
        except KeyError:
            self.counter[token] = 1 
        
    def train(self, *tokens):
        self.reset()
        for token in tokens:
            self.count(self.preprocess(token))
        self.load_counter()
    
    def load_counter(self):
        tokens = self.counter.keys() if self.max_words is None else \
            sorted(self.counter.keys(), key=self.counter.get, reverse=True)[:self.max_words]
        for token in tokens:
            self.add(token)
    
    def encode(self, tokens: List[str]) -> torch.Tensor:
        indices = []
        if self.bos_token:
            indices.append(self.bos_index)
        for token in tokens:
            try:
                indices.append(self.vocab[self.preprocess(token)])
            except KeyError:
                indices.append(self.unk_index)
        if self.eos_token:
            indices.append(self.eos_index)
        return torch.tensor(indices)
    
    def batch_encode(self, batch: List[List[str]]) -> torch.Tensor:
        return pad_sequence([self.encode(tokens) for tokens in batch], True, self.pad_index)
    
    def decode(self, indices: torch.Tensor, remove_special: Optional[str] = None) -> List[str]:
        tokens = [self.inv_vocab[index] for index in indices.tolist()]
        if isinstance(remove_special, str):
            tokens = [token if token not in self.specials else remove_special for token in tokens]
        elif remove_special is not None:
            tokens = [token for token in tokens if token not in self.specials]
        return tokens
    
    def batch_decode(self, batch: List[torch.Tensor], remove_special: Optional[str] = None) -> List[List[str]]:
        return [self.decode(indices, remove_special) for indices in batch]
    
    @property
    def pad_index(self) -> int:
        return self.vocab[self.pad_token]
    
    @property
    def unk_index(self) -> int:
        return self.vocab[self.unk_token]
    
    @property
    def bos_index(self) -> int:
        return self.vocab[self.bos_token]
    
    @property
    def eos_index(self) -> int:
        return self.vocab[self.eos_token]


    @property
    def special_indices(self) -> List[int]:
        return [self.vocab[token] for token in self.specials]