from __future__ import annotations
from torch.utils.data import Dataset 
from typing import List, Optional, Set
import os, json
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
        return int(self.IMG_PATH.split('_')[-1].replace('.jpg', ''))
        


class VQA(Dataset):
    def __init__(self, instances: List[Instance]):
        super().__init__()
        self.instances = instances
        
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]
        
    @classmethod
    def from_path(cls, img_folder: str, questions_path: str, annotations_path: str, max_images: int = int(1e3)) -> VQA:
        instances = [Instance(img_folder + '/' + path) for path in os.listdir(img_folder)][:max_images]
        instances = {instance.ID: instance for instance in instances}
        with open(questions_path, 'r') as file:
            questions = json.load(file)
        with open(annotations_path, 'r') as file:
            annotations = json.load(file)
        for question in questions['questions']:
            try:
                instances[question['image_id']].QUESTION = question['question']
            except KeyError:
                continue 
        for annotation in annotations['annotations']:
            try:
                instances[annotation['image_id']].ANSWER = annotation['multiple_choice_answer']
            except KeyError:
                continue 
        return VQA(list(instances.values()))
            
            
    @property
    def words(self) -> Set[str]:
        return set(flatten(instance.QUESTION.split() for instance in self.instances))

    @property
    def answers(self) -> Set[str]:
        return set(instance.ANSWER for instance in self.instances)
    
if __name__ == '__main__':
    train = VQA.from_path('VQA2/train2014', 'VQA2/train_questions.json', 'VQA2/train_annotations.json')

            
