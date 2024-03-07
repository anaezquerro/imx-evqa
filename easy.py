from __future__ import annotations
from torch.utils.data import Dataset 
from typing import List, Optional, Set
import os, json
from random import shuffle
from tokenizer import Tokenizer 
from fns import flatten


class Instance:
    def __init__(
        self,
        IMG_PATH: str, 
        QUESTION: Optional[str] = None, 
        ANSWER: Optional[str] = None,
        ):
        self.IMG_PATH = IMG_PATH
        self.QUESTION = QUESTION
        self.ANSWER = ANSWER
    
    @property
    def ID(self) -> int:
        return int(self.IMG_PATH.split('/')[-1].replace('.png', ''))
        

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
        instances = [Instance(img_folder + '/' + path) for path in os.listdir(img_folder)]
        instances = {instance.ID: instance for instance in instances}
        with open(questions_path, 'r') as file:
            questions = json.load(file)
        for question, answer, img_id in questions:
            try:
                instances[int(img_id)].QUESTION = question
                instances[int(img_id)].ANSWER = answer 
            except KeyError:
                continue 
        return EasyVQA(list(instances.values()))
            
            
    @property
    def words(self) -> List[str]:
        return flatten(instance.QUESTION.split() for instance in self.instances)

    @property
    def answers(self) -> List[str]:
        return [instance.ANSWER for instance in self.instances]
    
    def __iter__(self):
        return iter(self.instances)
    
    def drop(self, answer_tkz: Tokenizer, max_instances: Optional[int] = None) -> EasyVQA:
        instances = [instance for instance in self.instances if answer_tkz[instance.ANSWER] != answer_tkz.unk_index]
        if max_instances is not None:
            instances = instances[:max_instances]
        return EasyVQA(instances)
            
        
        
    

            
