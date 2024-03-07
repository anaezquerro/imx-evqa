from typing import Iterator
import torch 

def flatten(*lists) -> list:
    result = []
    for item in lists:
        if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, Iterator):
            result += flatten(*item)
        else:
            result.append(item)
    return result 



def to(device: str, *tensors):
    result = []
    for x in tensors:
        if isinstance(x, torch.Tensor):
            result.append(x.to(device))
        else:
            result.append(to(device, *x))
    return result 