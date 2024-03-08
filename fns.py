from typing import Iterator, Set
import torch 

def flatten(*lists) -> list:
    result = []
    for item in lists:
        if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, Iterator):
            result += flatten(*item)
        else:
            result.append(item)
    return result 


def fscore(pred: torch.Tensor, gold: torch.Tensor, binary: bool = True, exclude: Set[int] = set()):
    if binary:
        tp = (pred.flatten().to(torch.bool) & gold.flatten().to(torch.bool)).sum()
        prec, rec = tp/pred.sum(), tp/gold.sum()
        return (2*prec*rec)/(prec+rec+1e-12)
    else:
        classes = set(gold.unique().tolist()) - exclude
        fs = 0
        for c in classes:
            tp = ((pred.flatten() == c) & (gold.flatten() == c)).sum()
            prec, rec = tp/((pred.flatten() == c).sum() + 1e-12), tp/(gold.flatten() == c).sum()
            fs += (2*prec*rec)/(prec+rec+1e-12)
        fs /= (len(classes)+1e-12)
        return fs 





def to(device: str, *tensors):
    result = []
    for x in tensors:
        if isinstance(x, torch.Tensor):
            result.append(x.to(device))
        else:
            result.append(to(device, *x))
    return result 


def normalize(imgs: torch.Tensor):
    minx = imgs.view(imgs.shape[0], -1).min(-1)[0].view(-1, 1, 1, 1)
    maxx = imgs.view(imgs.shape[0], -1).max(-1)[0].view(-1, 1, 1, 1)
    return (imgs-minx)/(maxx-minx)
    