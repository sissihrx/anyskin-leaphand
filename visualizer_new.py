import rerun as rr
import numpy as np
import math
import time
from dynamixel_sdk import * 
import numpy as np
import argparse
import time
from anyskin import AnySkinProcess
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import sys
import pygame
from datetime import datetime
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


# addresses
ADDR_PROFILE_VELOCITY = 112 
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
DXL_ID = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
BAUDRATE = 3000000
PORT = '/dev/ttyUSB0'  
THRESHOLD = 30

def get_baseline():
    baseline_data = sensor_stream.get_data(num_samples=5)
    baseline_data = np.array(baseline_data)[:, 1:]
    baseline = np.mean(baseline_data, axis=0)
    return baseline

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
   
inscale = np.zeros(16)
inscale = torch.tensor(inscale, dtype=torch.float32)

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.inputs = self.data[:, 60:]
        self.outputs = self.data[:, :60]
        self.inputs = self.inputs / inscale
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 60)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()

def test(dataloader, model, loss_fn):
    model.eval()
    pval = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pval.append(pred.cpu().numpy())
    pval = np.concatenate(pval, axis = 0)
    # return (pval + 1) / 2 * 1200 - 600
    return pval

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
    parser = argparse.ArgumentParser(description="Test code to run a AnySkin streaming process in the background. Allows data to be collected without code blocking")
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", default="/dev/ttyACM0")
    parser.add_argument("-f", "--file", type=str, help="path to load data from", default=None)
    parser.add_argument("-v", "--viz_mode", type=str, help="visualization mode", default="3axis", choices=["magnitude", "3axis"])
    parser.add_argument("-s", "--scaling", type=float, help="scaling factor for visualization", default=7.0)
    parser.add_argument('-r', '--record', action='store_true', help='record data')
    args = parser.parse_args()

    file = None
    sensor_stream = AnySkinProcess(
            num_mags=20,
            port=args.port,
        )
    sensor_stream.start()
    time.sleep(3.0)
    baseline = get_baseline()

    rr.init("visualizer", spawn=True)
    rr.connect()

    portHandler = PortHandler(PORT)
    packetHandler = PacketHandler(2.0)

    if not portHandler.openPort():
        print("Failed to open port")
        quit()
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to set baudrate")
        quit()
    print("connected")

    model.load_state_dict(torch.load("pbmodel350.pth", map_location = 'cpu'))
    scale = np.loadtxt("pbscale350.txt")
    for i in range(16): inscale[i] = scale[i]

    shifta = np.zeros(60)
    shiftb = np.zeros(60)
    for k in range(1000):
        data = []
        
        # get 1 line of sensor data and store in data1
        sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
        sensor_data = sensor_data - baseline
        for x in range(60): data.append(sensor_data[x])
        
        for i in range(16):
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID[i], ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
            data.append(dxl_present_position)
        
        data1 = []
        data1.append(data)

        #get predicted vs actual for this position
        t1 = TextDataset(np.array(data1))
        td1 = DataLoader(t1, batch_size=1)
        a = gety(t1)
        b = test(t1, model, loss_fn)
        b = b + 7
        b = np.exp(b)
        b = b - 1308.9 - 1000
        
        #subtract a baseline so that a_0 = 0, b_0 = 0, a - b should be 0
        if k == 0:
            shifta = a.copy()
            shiftb = b.copy()
        a = a - shifta
        b = b - shiftb
        ans = (a - b).copy()
        
        
        rr.log(f"predicted", rr.Scalar(b[16]))
        rr.log(f"actual", rr.Scalar(a[16]))
        rr.log(f"subtracted", rr.Scalar(ans[16]))

    
    sensor_stream.pause_streaming()
    sensor_stream.join()
    
    for i in range(16):
        packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 0)

    portHandler.closePort() 