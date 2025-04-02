import rerun as rr
import numpy as np
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


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.inputs = self.data[:, 30:]
        self.outputs = self.data[:, :30]
        self.inputs = 2 * (self.inputs - 1800) / (3100 - 1800) - 1 
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


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
optimizer = torch.optim.SGD(model.parameters(), lr=1)

def test(dataloader, model, loss_fn):
    model.eval()
    pval = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pval.append(pred.cpu().numpy())
    pval = np.concatenate(pval, axis = 0)
    return (pval + 1) / 2 * 350 - 100

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

    rr.init("sensor_visualizer", spawn=True)
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

    model.load_state_dict(torch.load("2fmodels/2fmodelmore.pth", map_location = 'cpu'))
    
    #move fixed joints
    posrange = np.array([[1918, 1918], #here
                 [2400, 2900], 
                 [1800, 2600],
                 [2900, 3900],
                 [2058, 2058], #here
                 [2000, 2000], #
                 [1800, 1800],
                 [2900, 2900],
                 [2155, 2155], #here
                 [2000, 2000],
                 [1800, 1800],
                 [2000, 2000],
                 [1884, 1884], #here
                 [2100, 4000], #2100, 4000 for random
                 [2100, 2100], #here
                 [2200, 3100]
        ])
    for i in range (15):
        if i == 1 or i == 2 or i == 3 or i == 13: continue
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 1)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_PROFILE_VELOCITY, 60)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            quit()
        elif dxl_error != 0:
            print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
            quit()
    
    for i in range(15):
        if i == 1 or i == 2 or i == 3 or i == 13: continue
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_GOAL_POSITION, posrange[i][0])
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            failed = True
        elif dxl_error != 0:
            print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
           
    b = False
    while b == False:
        b = True
        for i in range(15):
            if i == 1 or i == 2 or i == 3 or i == 13: continue
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID[i], ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
            if abs(posrange[i][0] - dxl_present_position) > THRESHOLD:
                b = False

    print("moved")

    shift = np.zeros(30)
    for k in range(0):
        data = []
        
        # get sensor data
        sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
        sensor_data = sensor_data - baseline
        for x in range(30): data.append(sensor_data[x])
        
        for i in range(16):
            if i >= 4 and i < 12: continue
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID[i], ADDR_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
            data.append(dxl_present_position)
        
        data1 = []
        data1.append(data)

        t1 = TextDataset(data1)
        td1 = DataLoader(t1, batch_size=1)
        a = gety(t1)
        b = test(t1, model, loss_fn)
        if k == 0: shift = a - b
        ans = a - b - shift
        a = a - shift
        
        
        # rr.log(f"predicted", rr.Scalar(b[19]))
        rr.log(f"actual", rr.Scalar(a[21]))
        rr.log(f"subtracted", rr.Scalar(ans[21]))



    
    sensor_stream.pause_streaming()
    sensor_stream.join()
    
    for i in range(16):
        packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 0)

    portHandler.closePort() 