from __future__ import annotations
from torch.utils.data import Dataset 
from typing import List, Optional, Set
import os, json, cv2, torch 
from tokenizer import Tokenizer 
from fns import flatten
from random import shuffle
import numpy as np 

class Instance:
    def __init__(
        self,
        IMG_PATH: str, 
        QUESTION: str, 
        ANSWER: str
        ):
        self.IMG_PATH = IMG_PATH
        self.QUESTION = QUESTION.replace('?', ' ?')
        self.ANSWER = ANSWER
    
    @property
    def IMG(self) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(self.IMG_PATH), cv2.COLOR_BGR2RGB)
        
    @property 
    def INPUT(self) -> torch.Tensor:
        return torch.tensor(self.IMG/255).permute(2, 0, 1).to(torch.float32)
    
    @property 
    def MASK(self) -> torch.Tensor:
        img = cv2.imread(self.IMG_PATH, 0)
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[0]
        mask = img < thresh
        assert mask.mean() < 0.5, 'More than the 50 percent of the mask is an object'        
        return torch.tensor(mask)
    

    
    
        

class EasyVQA(Dataset):
    def __init__(self, instances: List[Instance]):
        super().__init__()
        self.instances = instances
        
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]
        
    @classmethod
    def from_path(cls, img_folder: str, questions_path: str) -> EasyVQA:
        inst_dict = {int(path.split('/')[-1].replace('.png', '')): img_folder + '/' + path for path in os.listdir(img_folder)}
        instances = []
        answers = set()
        with open(questions_path, 'r') as file:
            questions = json.load(file)
        for question, answer, img_id in questions:
            answers.add(answer)
            if answer in ['yes', 'no']:
                continue 
            try:
                instances.append(Instance(inst_dict[int(img_id)], question, answer))
            except KeyError:
                continue 
        # dict_inst = {token: [] for token in answers}
        # for instance in instances:
        #     dict_inst[instance.ANSWER].append(instance)
        # medium = int(np.mean([len(inst) for answer, inst in dict_inst.items() if answer not in ['yes', 'no']]))
        # dict_inst['yes'] = dict_inst['yes'][:medium]
        # dict_inst['no'] = dict_inst['no'][:medium]
        # instances = flatten(*dict_inst.values())
        return EasyVQA(instances)
            
            
    @property
    def words(self) -> List[str]:
        return flatten(instance.QUESTION.split() for instance in self.instances)

    @property
    def answers(self) -> List[str]:
        return [instance.ANSWER for instance in self.instances]
    
    def __iter__(self):
        return iter(self.instances)

        
    

            
