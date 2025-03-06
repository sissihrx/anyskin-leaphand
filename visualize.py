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
        self.outputs -= self.outputs[0]
        self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
        # self.outputs = 2 * (self.outputs + 1000) / (2000) - 1 #scale iwth -1000 to 1000, 1800 - 3100


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 30)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    pval = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pval.append(pred.cpu().numpy())
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches * 100
    pval = np.concatenate(pval, axis = 0)

    print(f"Test Error: \n Avg loss: {test_loss} \n")
    return (pval + 1) * 650 / 2 - 250

def gety(dataloader):
    yval = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.to(device)
            y = y.cpu().numpy()
            yval.append(y)
    yval = np.concatenate(yval, axis = 0)
    return yval



if __name__ == "__main__":
    test_data = TextDataset("2fingermodel/2fmodeldata/2fnocontactdata.txt")
    test_dataloader = DataLoader(test_data, batch_size=100)

    model.load_state_dict(torch.load("2fingermodel/2fmodels/2fmodelact1.pth"))
    res = test(test_dataloader, model, loss_fn)
    act = gety(test_dataloader)
    res = res.T
    act = act.T

    t = np.zeros(len(res[0]))
    for i in range(len(res[0])):
        t[i] = i

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7))
    n = 15
    for i in range(3):
        for j in range(5):
            axes[i][j].plot(t, res[n], label = "predicted", linestyle="-.")
            axes[i][j].plot(t, act[n], label = "actual", linestyle="-")
            n += 1
            plt.legend()

    fig.canvas.draw()
    for i in range(3):
        for j in range(5):
            axes[i][j].set_ylim(min(axes[i][j].get_ylim()[0], -50), max(axes[i][j].get_ylim()[1], 50))

    plt.tight_layout()
    plt.show()


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch import nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import datasets
# from torchvision.transforms import ToTensor


# device = (
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps"
#         if torch.backends.mps.is_available()
#         else "cpu"
#     )


# class TextDataset(Dataset):
#     def __init__(self, file_path):
#         self.data = []
#         with open(file_path, "r") as file:
#             for line in file:
#                 self.data.append([float(x) for x in line.strip().split(" ")])
#         self.data = torch.tensor(self.data, dtype=torch.float32)
        
#         self.inputs = self.data[:, 60:]
#         self.outputs = self.data[:, :60]
#         self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
#         self.outputs = 2 * (self.outputs + 1000) / (2000) - 1 #scale iwth -1000 to 1000, 1800 - 3100


#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.outputs[idx]


# # Define model
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(16, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, 60)
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork().to(device)
# print(model)

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss = 0
#     pval = []
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             pval.append(pred.cpu().numpy())
#             test_loss += loss_fn(pred, y).item()
#     test_loss /= num_batches * 100
#     pval = np.concatenate(pval, axis = 0)

#     print(f"Test Error: \n Avg loss: {test_loss} \n")
#     return pval

# def gety(dataloader):
#     yval = []
#     with torch.no_grad():
#         for X, y in dataloader:
#             y = y.to(device)
#             y = y.cpu().numpy()
#             yval.append(y)
#     yval = np.concatenate(yval, axis = 0)
#     return yval



# if __name__ == "__main__":
#     test_data = TextDataset("fullmodeldata/fulltestdata.txt")
#     test_dataloader = DataLoader(test_data, batch_size=100)

#     temp = TextDataset("fullmodeldata/test.txt")
#     tempd = DataLoader(temp, batch_size=100)

#     model.load_state_dict(torch.load("fullmodelrs.pth"))
#     res = test(test_dataloader, model, loss_fn)
#     act = gety(test_dataloader)
#     temp = gety(tempd)
#     res += act[0] - temp[0]

#     res = (res + 1) * 2000 / 2 - 1000
#     act = (act + 1) * 2000 / 2 - 1000
#     res = res.T
#     act = act.T
    
#     t = np.zeros(len(res[0]))
#     for i in range(len(res[0])):
#         t[i] = i

#     fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7))
#     n = 15
#     for i in range(3):
#         for j in range(5):
#             axes[i][j].plot(t, res[n], label = "predicted", linestyle="-.")
#             axes[i][j].plot(t, act[n], label = "actual", linestyle="-")
#             n += 1
#             plt.legend()
#     plt.tight_layout()
#     plt.show()