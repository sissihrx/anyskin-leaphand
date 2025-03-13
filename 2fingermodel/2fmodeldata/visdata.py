import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


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
        self.outputs = self.outputs - self.outputs[0]
        
        self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
        # self.outputs = 2 * (self.outputs + 100) / 350 - 1 #scale iwth -100 to 250, 1800 - 3100
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def gety(dataloader):
    yval = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            y = y.cpu().numpy()
            yval.append(y)
    yval = np.concatenate(yval, axis = 0)
    return yval

def getx(dataloader):
    yval = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            X = X.cpu().numpy()
            yval.append(X)
    yval = np.concatenate(yval, axis = 0)
    return yval



if __name__ == "__main__":
    t1 = TextDataset("2fingermodel/2fmodeldata/test1.txt")#503
    td1 = DataLoader(t1, batch_size=100)
    t2 = TextDataset("2fingermodel/2fmodeldata/testtest.txt")#503
    td2 = DataLoader(t2, batch_size=100)
    t3 = TextDataset("2fingermodel/2fmodeldata/test3.txt")#503
    td3= DataLoader(t3, batch_size=100)
    t4 = TextDataset("2fingermodel/2fmodeldata/test4.txt")#503
    td4 = DataLoader(t4, batch_size=100)
    t5 = TextDataset("2fingermodel/2fmodeldata/test5.txt")#503
    td5 = DataLoader(t5, batch_size=100)
    t6 = TextDataset("2fingermodel/2fmodeldata/test6.txt")#503
    td6 = DataLoader(t6, batch_size=100)
    t7 = TextDataset("2fingermodel/2fmodeldata/test7.txt")#503
    td7 = DataLoader(t7, batch_size=100)
    t8 = TextDataset("2fingermodel/2fmodeldata/2fnocontactdata.txt")#503
    td8 = DataLoader(t8, batch_size=100)

    r1 = gety(td1).T
    r2 = gety(td2).T
    r3 = gety(td3).T
    r4 = gety(td4).T
    r5 = gety(td5).T
    r6 = gety(td6).T
    r7 = gety(td7).T
    r8 = gety(td8).T

    t = np.zeros(len(r1[0]))
    for i in range(len(r1[0])):
        t[i] = i

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7))
    n = 15
    for i in range(3):
        for j in range(5):
            axes[i][j].plot(t, r1[n], linestyle="-.")
            axes[i][j].plot(t, r2[n], linestyle="--")
            # axes[i][j].plot(t, r3[n], linestyle="-")
            # axes[i][j].plot(t, r4[n], linestyle=":")
            # axes[i][j].plot(t, r5[n], linestyle="-")
            # axes[i][j].plot(t, r6[n], linestyle="-")
            # axes[i][j].plot(t, r7[n], linestyle="-")
            # axes[i][j].plot(t, r8[n], linestyle="-")
            n += 1
            plt.legend()

    fig.canvas.draw()
    # for i in range(3):
    #     for j in range(5):
    #         axes[i][j].set_ylim(min(axes[i][j].get_ylim()[0], -50), max(axes[i][j].get_ylim()[1], 50))

    plt.tight_layout()
    plt.savefig("plotsold")
