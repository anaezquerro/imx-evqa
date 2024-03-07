from __future__ import annotations
import torch, cv2, pickle
from torch import nn 
from data import VQA, Instance
from typing import Tuple, List, Optional
from model import VQAModel
from tokenizer import Tokenizer
from transformers import AutoImageProcessor
from fns import to 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchvision.transforms import Resize
from torch.optim import AdamW, Optimizer

class XVQASystem:
    IMG_SIZE = (320, 480)
    PARAMS = ['img_size']
    
    def __init__(
        self, 
        model: nn.Module, 
        word_tkz: Tokenizer, 
        answer_tkz: Tokenizer,
        img_size: Optional[Tuple[int, int]] = None,
        device: str = 'cuda:0'
    ):
        self.model = model 
        self.word_tkz = word_tkz
        self.answer_tkz = answer_tkz
        self.img_size = img_size if img_size is not None else XVQASystem.IMG_SIZE
        self.device = device
        self.resize = Resize(img_size)
        
    def train(
        self,
        train: VQA,
        val: VQA,
        epochs: int,
        batch_size: int,
        lr: float = 1e-3,
        max_norm: float = 5.0
    ):
        train_dl = DataLoader(train, batch_size=batch_size, collate_fn=self.transform, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr)
        
        for epoch in range(epochs):
            with tqdm(total=len(train_dl), desc='train') as bar:
                for words, imgs, answers in train_dl:
                    s_answer, mask = self.model(imgs, words) 
                    active_ratio = mask.sum()/torch.prod(torch.tensor(mask.shape))
                    loss = self.model.loss(s_answer, answers).to(self.device) 
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=2)
                    optimizer.step()
                    bar.update(1)
                    train_acc = ((s_answer.argmax(-1).to(self.device) == answers)*1.0).mean()*100
                    bar.set_postfix({'loss': f'{loss.item():.2f}', 'ratio': f'{active_ratio.item():.2f}', 'train_acc': f'{train_acc:.2f}'})
            acc = self.evaluate(val, batch_size)
            print(f'Epoch {epoch+1}: loss={loss.item():.2f}, ratio={active_ratio.item():.2f}, train_acc: {train_acc:.2f}, val_acc={acc:.2f}')
            
    @torch.no_grad()    
    def evaluate(
        self,
        val: VQA,
        batch_size: int
    ) -> float:
        val_dl = DataLoader(val, batch_size=batch_size, collate_fn=self.transform)
        acc = 0
        for words, imgs, answers in tqdm(val_dl, total=len(val_dl), desc='eval'):
            s_answer, _ = self.model(imgs, words)
            acc += (s_answer.argmax(-1).to(self.device) == answers).sum()
        return float(acc/len(val))*100
    
            

    def transform(self, batch: List[Instance]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        questions = self.word_tkz.batch_encode([instance.QUESTION.split() for instance in batch])
        answers = self.answer_tkz.encode([instance.ANSWER for instance in batch])
        imgs = torch.stack([self.resize(torch.tensor(cv2.imread(instance.IMG_PATH)).permute(2, 0, 1)) for instance in batch], 0)/255
        return to(self.device, questions, imgs, answers)
    
    
    def save_model(self, path: str, optimizer: Optimizer):
        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
        
    def save(self, path: str):
        self.word_tkz.save(f'{path}/{self.word_tkz.field}.tkz')
        self.answer_tkz.save(f'{path}/{self.answer_tkz.field}.tkz')
        params = {param: getattr(self, param) for param in self.PARAMS}
        with open(f'{path}/params.pickle', 'wb') as writer:
            pickle.dump(params, writer)
        

    
    @classmethod
    def build(
        cls, 
        data: VQA,
        word_embed_size: int, 
        devices: Tuple[str,str],
        img_size: Optional[Tuple[int, int]] = None,
        max_targets: int = 200
    ) -> XVQASystem:
        answer_tkz = Tokenizer('ANSWER', max_words=max_targets)
        answer_tkz.train(*data.answers)
        data = data.drop(answer_tkz)
        word_tkz = Tokenizer('WORD', lower=True)
        word_tkz.train(*data.words)
        img_size = img_size if img_size is not None else XVQASystem.IMG_SIZE
        model = VQAModel(img_size, word_embed_size, len(word_tkz), word_tkz.pad_index, len(answer_tkz), devices)
        return XVQASystem(model, word_tkz, answer_tkz, img_size, devices[0])
        
        
if __name__ == '__main__':
    train = VQA.from_path('VQA2/train2014/', 'VQA2/train_questions.json', 'VQA2/train_annotations.json')
    val = VQA.from_path('VQA2/val2014/', 'VQA2/val_questions.json', 'VQA2/val_annotations.json')
    system = XVQASystem.build(train, 200, ('cuda:0', 'cuda:1'))
    train = train.drop(system.answer_tkz, int(5e3))
    val = val.drop(system.answer_tkz, int(5e2))
    system.train(train, val, epochs=100, batch_size=30)