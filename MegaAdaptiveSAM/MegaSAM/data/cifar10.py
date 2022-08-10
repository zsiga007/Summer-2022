import torch
from torchvision import datasets, transforms

def cifar10_data_gen(N, batch_size):
    tr_split_len = 0.8 * N
    te_split_len = 0.2 * N

    tr = datasets.CIFAR10(
        root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
    te = datasets.CIFAR10(
        root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
    cifar10_train = torch.utils.data.random_split(tr, [tr_split_len, len(tr)-tr_split_len])[0]
    cifar10_test = torch.utils.data.random_split(te, [te_split_len, len(te)-te_split_len])[0]
    train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, cifar10_test