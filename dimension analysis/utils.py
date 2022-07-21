import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms

def effective_rank(tensor):
    _, s,_ = torch.svd(tensor.float())
    norm = torch.sum(torch.abs(s))
    entropy = 0
    for sv in list(s):
        sv = float(sv)
        entropy -= sv * np.log(sv/norm)/norm
    return np.exp(entropy)
# Testing
# tensor = torch.tensor([[1,  0],[0,  1]])
# print(effective_rank(tensor))

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def flatten_and_together(list_of_tensors):
    new_list = []
    for tensor in list_of_tensors:
        new_list.append(torch.flatten(tensor))
    return torch.cat(new_list)
# Testing
# print(flatten_and_together([torch.tensor([[1],[2]]), torch.tensor([1,2])]))
