from __future__ import annotations
import torch, cv2, pickle, shutil, os
from torch import nn 
from easy import EasyVQA, Instance
from typing import Tuple, List, Dict
from model import VQAModel
from tokenizer import Tokenizer
from transformers import AutoImageProcessor
from fns import to, fscore
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torchvision.transforms import Compose, GaussianBlur, RandomHorizontalFlip, RandomVerticalFlip
from torch.optim import Adam, Optimizer
import numpy as np
from scipy import ndimage 
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
        lr: float = 1e-4,
        dev_patience: int = 50,
    ):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        train_dl = DataLoader(train, batch_size=batch_size, collate_fn=self.transform, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr)
        best_fs, improv = 0, dev_patience
        for epoch in range(epochs):
            with tqdm(total=len(train_dl), desc='train') as bar:
                for words, imgs, answers, masks in train_dl:
                    s_answer, s_mask = self.model(imgs, words) 
                    n_comp = sum(ndimage.label(m.detach().cpu().numpy())[1] for m in s_mask.unbind(0))/s_mask.shape[0]
                    loss = self.model.loss(s_answer, s_mask, answers, masks)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    bar.update(1)
                    cls_fs = fscore(s_answer.argmax(-1), answers, False)
                    mask_fs = fscore(s_mask > 0, masks)
                    exc_fs = fscore(s_answer.argmax(-1), answers, False, {self.answer_tkz['yes'], self.answer_tkz['no']})
                    bar.set_postfix({'loss': f'{loss.item():.2f}', 'n_comp': n_comp, 'cls_fs': f'{cls_fs*100:.2f}', 'mask_fs': f'{mask_fs*100:.2f}', 'exc_fs': f'{exc_fs*100:.2f}'})
            cls_fs, mask_fs, exc_fs = self.evaluate(val, batch_size)
            print(f'Epoch {epoch+1}: loss={loss.item():.2f}, n_components={n_comp}, cls_fs={cls_fs*100:.2f}, mask_fs={mask_fs*100:.2f}, exc_fs={exc_fs*100:.2f}' \
                + (' (improved)' if cls_fs > best_fs else ''))
            
            if cls_fs > best_fs:
                best_fs = cls_fs
                improv = dev_patience
                self.save(path)
            else:
                improv -= 1
                
            if improv == 0: 
                print('No improvement in the validation set')
                break 
                
        self.save(path)
            
    @torch.no_grad()    
    def evaluate(
        self,
        val: EasyVQA,
        batch_size: int
    ) -> Tuple[float, float, float]:
        val_dl = DataLoader(val, batch_size=batch_size, collate_fn=self.transform)
        cls_fs, mask_fs, exc_fs = 0, 0, 0
        for words, imgs, answers, masks in tqdm(val_dl, total=len(val_dl), desc='eval'):
            s_answer, s_mask = self.model(imgs, words)
            cls_fs += fscore(s_answer.argmax(-1), answers, False)
            exc_fs += fscore(s_answer.argmax(-1), answers, False, {self.answer_tkz['yes'], self.answer_tkz['no']})
            mask_fs += fscore(s_mask > 0, masks)
        return map(lambda x: float(x/len(val_dl)), (cls_fs, mask_fs, exc_fs))
    
    @torch.no_grad()
    def predict(self, test: EasyVQA, batch_size: int) -> Tuple[List[Dict[str, float]], List[torch.Tensor]]:
        test_dl = DataLoader(test, batch_size=batch_size, collate_fn=self.transform)
        answers, masks = [], []
        for words, imgs, _, _ in tqdm(test_dl, total=len(test_dl), desc='predict'):
            s_answer, s_mask = self.model(imgs, words)
            answers += s_answer.unbind(0)
            masks += s_mask.view(imgs.shape[0], imgs.shape[-2], imgs.shape[-1]).unbind(0)
        answers = [
            {self.answer_tkz.inv_vocab[index]: value for index, value in enumerate(answer.tolist()) 
             if index not in self.answer_tkz.special_indices} 
            for answer in answers
        ]
        return answers, masks 
    
    
    @torch.no_grad()
    def feedback(self, test: EasyVQA, masks: List[torch.Tensor], batch_size: int) -> List[Dict[str, float]]:
        test_dl = DataLoader(test, batch_size=batch_size, collate_fn=self.transform)
        answers, masks = [], masks.copy()
        for words, imgs, _, _ in tqdm(test_dl, total=len(test_dl), desc='feedback'):
            mask = torch.stack([masks.pop(0) > 0 for _ in range(words.shape[0])]).unsqueeze(1)
            s_answer = self.model.feedback(imgs, words, mask)
            answers += s_answer.unbind(0)
        answers = [
            {self.answer_tkz.inv_vocab[index]: value for index, value in enumerate(answer.tolist()) 
             if index not in self.answer_tkz.special_indices} 
            for answer in answers
        ]
        return answers
    
    def transform(self, batch: List[Instance]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        questions = self.word_tkz.batch_encode([instance.QUESTION.split() for instance in batch])
        answers = self.answer_tkz.encode([instance.ANSWER for instance in batch])
        imgs = torch.stack([instance.INPUT for instance in batch], 0)
        masks = torch.stack([instance.MASK for instance in batch], 0)
        return to(self.device, questions, imgs, answers, masks)
    
    
    def save(self, path: str):
        self.word_tkz.save(f'{path}/{self.word_tkz.field}.tkz')
        self.answer_tkz.save(f'{path}/{self.answer_tkz.field}.tkz')
        params = dict(word_embed_size=self.model.word_embed_size, img_size=self.model.img_size)
        with open(f'{path}/params.pickle', 'wb') as writer:
            pickle.dump(params, writer)
        torch.save({'model': self.model.state_dict()}, f'{path}/model.pt')

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
        model = VQAModel(
            params['img_size'], params['word_embed_size'], len(word_tkz), word_tkz.pad_index, 
            len(answer_tkz), weights=answer_tkz.weights
        ).to(device)
        dummy = (torch.empty(1, 3, *params['img_size']), torch.randint(0, len(word_tkz), (1, 10)))
        model.forward(*to(device, *dummy))
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
        word_tkz = Tokenizer('WORD', lower=True)
        word_tkz.train(*data.words)
        model = VQAModel(
            img_size, word_embed_size, len(word_tkz), word_tkz.pad_index, len(answer_tkz), 
            weights=answer_tkz.weights
        ).to(device)
        return XVQASystem(model, word_tkz, answer_tkz, device)
        
        
if __name__ == '__main__':
    train = EasyVQA.from_path('easy-vqa/train/images/', 'easy-vqa/train/questions.json')
    val = EasyVQA.from_path('easy-vqa/test/images/', 'easy-vqa/test/questions.json')
    system = XVQASystem.build(train, 200, device='cuda:0')
    system.train(train, val, 'results/', epochs=500, batch_size=500)