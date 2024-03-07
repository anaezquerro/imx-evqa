from __future__ import annotations
import torch, cv2, pickle
from torch import nn 
from easy import EasyVQA, Instance
from typing import Tuple, List, Optional
from model import VQAModel
from tokenizer import Tokenizer
from transformers import AutoImageProcessor
from fns import to 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchvision.transforms import Resize
from torch.optim import AdamW, Optimizer

IMG_SIZE = (64, 64)


class XVQASystem:
    
    def __init__(
        self, 
        model: nn.Module, 
        word_tkz: Tokenizer, 
        answer_tkz: Tokenizer,
        device: str = 'cuda:0'
    ):
        self.model = model 
        self.word_tkz = word_tkz
        self.answer_tkz = answer_tkz
        self.device = device
        
    def train(
        self,
        train: EasyVQA,
        val: EasyVQA,
        path: str,
        epochs: int = 20,
        batch_size: int = 50,
        lr: float = 1e-4
    ):
        train_dl = DataLoader(train, batch_size=batch_size, collate_fn=self.transform, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr)
        
        best_acc = 0
        for epoch in range(epochs):
            with tqdm(total=len(train_dl), desc='train') as bar:
                for words, imgs, answers in train_dl:
                    s_answer, mask = self.model(imgs, words) 
                    active_ratio = mask.sum()/torch.prod(torch.tensor(mask.shape))
                    loss = self.model.loss(s_answer, answers) + active_ratio
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    bar.update(1)
                    train_acc = ((s_answer.argmax(-1).to(self.device) == answers)*1.0).mean()*100
                    bar.set_postfix({'loss': f'{loss.item():.2f}', 'ratio': f'{active_ratio.item():.2f}', 'train_acc': f'{train_acc:.2f}'})
            acc = self.evaluate(val, batch_size)
            print(f'Epoch {epoch+1}: loss={loss.item():.2f}, ratio={active_ratio.item():.2f}, train_acc: {train_acc:.2f}, val_acc={acc:.2f}')
            
            if acc > best_acc:
                best_acc = acc 
                self.save_model(f'{path}/model.pt', optimizer)
        self.save(path)
            
    @torch.no_grad()    
    def evaluate(
        self,
        val: EasyVQA,
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
        imgs = torch.stack([torch.tensor(cv2.imread(instance.IMG_PATH))/255 for instance in batch], 0).permute(0, 3, 1, 2)
        return to(self.device, questions, imgs, answers)
    
    
    def save_model(self, path: str):
        torch.save({'model': self.model.state_dict()}, path)
        
    def save(self, path: str):
        self.word_tkz.save(f'{path}/{self.word_tkz.field}.tkz')
        self.answer_tkz.save(f'{path}/{self.answer_tkz.field}.tkz')
        params = dict(word_embed_size=self.model.word_embed_size, img_size=self.model.img_size)
        with open(f'{path}/params.pickle', 'wb') as writer:
            pickle.dump(params, writer)
        
    @classmethod
    def load(
        cls, 
        path: str,
        device: str 
    ) -> XVQASystem:
        with open(f'{path}/params.pickle', 'rb') as reader:
            params = pickle.load(reader)
        word_tkz = Tokenizer.load(f'{path}/WORD.tkz')
        answer_tkz = Tokenizer.load(f'{path}/ANSWER.tkz')
        model = VQAModel(params.pop('img_size'), params.pop('word_embed_size'), len(word_tkz), word_tkz.pad_index, len(answer_tkz)).to(device)
        model.load_state_dict(torch.load(f'{path}/model.pt')['model'])
        return XVQASystem(model, word_tkz, answer_tkz, device)
        
    
    @classmethod
    def build(
        cls, 
        data: EasyVQA,
        word_embed_size: int, 
        device: str,
        img_size: Tuple[int, int] = IMG_SIZE
    ) -> XVQASystem:
        answer_tkz = Tokenizer('ANSWER')
        answer_tkz.train(*data.answers)
        data = data.drop(answer_tkz)
        word_tkz = Tokenizer('WORD', lower=True)
        word_tkz.train(*data.words)
        model = VQAModel(img_size, word_embed_size, len(word_tkz), word_tkz.pad_index, len(answer_tkz)).to(device)
        return XVQASystem(model, word_tkz, answer_tkz, device)
        
        
if __name__ == '__main__':
    train = EasyVQA.from_path('easy-vqa/train/images/', 'easy-vqa/train/questions.json')
    val = EasyVQA.from_path('easy-vqa/test/images/', 'easy-vqa/test/questions.json')
    system = XVQASystem.build(train, 50, device='cuda:0')
    system.train(train, val, epochs=100, batch_size=40)