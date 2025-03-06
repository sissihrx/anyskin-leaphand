import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r") as file:
            for line in file:
                self.data.append([float(x) for x in line.strip().split(" ")])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, 30:]
        self.outputs = self.data[:, :30]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def getx(dataloader):
    xval = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            X = X.cpu().numpy()
            xval.append(X)
    xval = np.concatenate(xval, axis = 0)
    return xval

if __name__ == "__main__":
    training_data = TextDataset("2fingermodel/2fmodeldata/2frandcontdata.txt")
    train_dataloader = DataLoader(training_data, batch_size=100)

    test_data = TextDataset("2fingermodel/2fmodeldata/2fnocontactdata.txt")
    test_dataloader = DataLoader(test_data, batch_size=100)
    
    yval = getx(train_dataloader)
    thumbpos = np.array([1884, 3050, 2100, 2650])
    j1 = []
    j2 = []
    j3 = []

    for row in yval:
        diff = 0
        for i in range(4): diff += abs(row[4 + i] - thumbpos[i])
        if diff / 4 <= 40:
            j1.append(row[1])
            j2.append(row[2])
            j3.append(row[3])
    
    print(len(j1))

    y1 = getx(test_dataloader)
    x1 = []
    x2 = []
    x3 = []
    for row in y1:
        x1.append(row[1])
        x2.append(row[2])
        x3.append(row[3])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(j1, j2, j3, color = 'b', label = "training data")
    ax.scatter(x1, x2, x3, color = 'r', label = "test data")

    ax.set_xlabel('Index joint 1 position')
    ax.set_ylabel('Joint 2 position')
    ax.set_zlabel('Joint 3 position')


    plt.show()
    
        