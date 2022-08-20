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

def min_regrouping(C):
    """
    Returns a list of lists containing the pairs of batches that should be 
    put together.
    """
    n, _ = C.shape
    final = []
    taken = []
    fill = C.max() + 1
    indices = np.arange(0,n)
    np.random.shuffle(indices)
    for row in indices:
        if row in taken:
            continue
        min = np.argmin(C[row, :])
        while min in taken or min == row:
            C[row, min] = fill
            min = np.argmin(C[row, :])
        final.append([row, min])
        taken.append(min)
    return final

def max_regrouping(C):
    """
    Returns a list of lists containing the pairs of batches that should be 
    put together.
    """
    n, _ = C.shape
    final = []
    taken = []
    fill = C.min() -1
    indices = np.arange(0,n)
    np.random.shuffle(indices)
    for row in indices:
        if row in taken:
            continue
        max = np.argmax(C[row, :])
        while max in taken or max == row:
            C[row, max] = fill
            max = np.argmax(C[row, :])
        final.append([row, max])
        taken.append(max)
    return final

# C = np.random.rand(4,4)
# print(C)
# print(regrouping(C))

def merge_loader(l, shuffle, batch_size=10):
    new_loader_list = []
    for pair in shuffle:
        # print(l[pair[1]][0])
        new_loader_list.append([torch.cat((l[pair[0]][0], l[pair[1]][0])), 
                                torch.cat((l[pair[0]][1], l[pair[1]][1]))])
        
    finallist = []
    for i in range(len(new_loader_list)):
        for k in range(batch_size):
            finallist.append([new_loader_list[i][0][k], new_loader_list[i][1][k]])
    return finallist
