import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets

def moons_data_gen(N=1000, batch_size=1000):
    all_data = []
    X,y = datasets.make_moons(n_samples=N, shuffle=True, noise=0.2, random_state=1234)
    y = np.reshape(y, (len(y),1))
    for index, point in enumerate(X):
        point=np.reshape(point, (2,1))
        all_data.append([torch.tensor(point), y[index][0]])
    zeros_x = [all_data[i][0][0] for i in range(N) if not all_data[i][1]]
    zeros_y = [all_data[i][0][1] for i in range(N) if not all_data[i][1]]

    ones_x = [all_data[i][0][0] for i in range(N) if all_data[i][1]]
    ones_y = [all_data[i][0][1] for i in range(N) if all_data[i][1]]

    #plt.plot(zeros_x, zeros_y, '.')
    #plt.plot(ones_x, ones_y, '.')

    train_data = all_data[:N*8//10]
    test_data = all_data[N*8//10:]
    batch_size = 1000
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    
    return all_data, train_loader, test_loader, test_data
  