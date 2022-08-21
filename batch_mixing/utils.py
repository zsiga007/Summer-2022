import numpy as np
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
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

def train_multi_model(model, train_data, test_data, optim='SGD', batch_size=200, epochs=10, tracking=False,
                shuffle_loader=True, lr=0.01, momentum=0.9, criterion=nn.CrossEntropyLoss()):
    """
    Trains a model for classification.

    Inputs:
    model: an intantiation of the class of the model we want to train.
    tracking: tracks test accuracy during training, Boolean
    optimizer: String, can be 'SGD', 'SAM', 'Adam', MegaSAM
    rho: float, the rho parameter for SAM
    """
    from torchvision import datasets, transforms
    from tqdm.notebook import tqdm, trange
    num_of_params = get_n_params(model)

    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_loader)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(test_data, shuffle=False)

    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=lr)

    training_losses = []
    training_accuracies = []
    validation_accuracies = []
    C_matrices = []
    train_loader2 = torch.utils.data.DataLoader(train_data, batch_size=batch_size//2, shuffle=True) 

    # Iterate through train set minibatchs
 
    for epoch in trange(epochs):  
        per_epoch_loss = 0
        correct = 0
        row_counter = 0
        tensor = torch.zeros((len(train_loader2), num_of_params))
        for points2, labels2 in train_loader2:
            optimizer2.zero_grad()
            x2 = points2[:, None]
            x2 = x2.to(device)
            labels2 = labels2.to(device)[:, None]
            z = model(x2.float())
            y2 = z.reshape((z.shape[0],10,1))
            y2 = y2.double()
            labels2 = labels2.long()
            loss2 = criterion(y2, labels2)
            loss2.backward()
            # optimizer2.step()
            tensor[row_counter, :] = flatten_and_together([param.grad for param in list(model.parameters())])
            row_counter += 1
        C_matrix = tensor @ tensor.T
        C_matrix = C_matrix.detach().numpy()
        new_shuffle = max_regrouping(C_matrix)
        new_loader_list = merge_loader(list(train_loader2), new_shuffle, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(new_loader_list, batch_size=batch_size, shuffle=False)
        # e = 0
        # for i in train_loader:
        #     print(i)
        #     e+=1
        #     if e==1:
        #         break
        # C_matrices.append(C_matrix)
        model.zero_grad()
        for numbers, labels in train_loader:
            x = numbers[:,None]
            x = x.to(device)
            labels = labels.to(device)[:,None]
            # Zero out the gradients
            optimizer.zero_grad()
            # Forward pass
            z = model(x.float())
            y = z.reshape((z.shape[0],10,1))

            y = y.double()
            labels = labels.long()
            loss = criterion(y, labels)

            loss.backward()
            optimizer.step()

            if tracking:
                # Tracking loss
                per_epoch_loss += loss
                # Train accuracy tracking
                predictions = torch.argmax(y, dim=1)
                correct += torch.sum((predictions == labels).float())
                # print(torch.sum((predictions == labels).float()))
 

        if tracking:
            correct_test = 0
            with torch.no_grad():
                    # Iterate through test set minibatchs 
                    for numbers2, labels_ in val_loader:
                        numbers2 = numbers2.to(device)
                        labels_ = labels_.double().to(device)[:,None]
                        # Forward pass
                        x2 = numbers2[:,None]
                        y2 = model(x2)
                        predictions2 = torch.argmax(y2)
                        correct_test += torch.sum((predictions2 == labels_).float())

            training_losses.append(per_epoch_loss/len(train_loader))
            training_accuracies.append(correct/len(train_data))
            validation_accuracies.append(correct_test/len(test_data))
    training_losses = [i.item() for i in training_losses]
    training_accuracies = [i.item() for i in training_accuracies]
    validation_accuracies = [i.item() for i in validation_accuracies]

    return model, training_losses, training_accuracies, validation_accuracies, optimizer
