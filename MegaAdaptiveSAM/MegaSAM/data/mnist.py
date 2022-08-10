import torch
from torchvision import datasets, transforms


def mnist_data_gen():
    # tr_split_len = 0.8 * N
    # te_split_len = 0.2 * N
  
    training_set = datasets.MNIST(
        root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(
        root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
    # mnist_train = torch.utils.data.random_split(tr, [tr_split_len, len(tr)-tr_split_len])[0]
    # mnist_test = torch.utils.data.random_split(te, [te_split_len, len(te)-te_split_len])[0]
    # train_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(te, batch_size=batch_size, shuffle=True)
    return training_set, test_set
