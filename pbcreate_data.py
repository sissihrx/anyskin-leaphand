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


def get_baseline():
    baseline_data = sensor_stream.get_data(num_samples=5)
    baseline_data = np.array(baseline_data)[:, 1:]
    baseline = np.mean(baseline_data, axis=0)
    return baseline


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
    filename = "fullmodeldata/pbdata2"
    pygame.init()
    time.sleep(0.1)
    baseline = get_baseline()

    ADDR_PROFILE_VELOCITY = 112 
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    DXL_ID = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    BAUDRATE = 3000000
    PORT = '/dev/ttyUSB0'  
    THRESHOLD = 30

    portHandler = PortHandler(PORT)
    packetHandler = PacketHandler(2.0)

    if not portHandler.openPort():
        print("Failed to open port")
        quit()
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to set baudrate")
        quit()

    
    for i in range (16):
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 1)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_PROFILE_VELOCITY, 60)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
            quit()
        elif dxl_error != 0:
            print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
            quit()

    print("connected")

    posrange = np.array([[1448, 2500], 
                 [1830, 3530], 
                 [1730, 3230],
                 [2830, 4250],
                 [1448, 2500], 
                 [1810, 3520], 
                 [1750, 3190],
                 [2850, 4280],
                 [1448, 2500], 
                 [1870, 3500], 
                 [1750, 3180],
                 [1800, 3150],
                 [1770, 3450], 
                 [2030, 3250], #13
                 [1230, 3280], 
                 [1250, 3300]
        ])
    # 8/9, 4/5, 0/1 swapped
    # flipped small/large: 12, 8, 4, 0
    
    firstpos = np.array([1918, 2662, 2427, 3886, 2058, 2595, 2010, 3600, 2155, 2536, 2250, 2444, 1884, 3466, 2100, 2594])
    positions = np.loadtxt("pb_simulation/positionstr.txt")
    
    #translate positions from normalized 0-1 to the range of joints, fix flipped angles in pb simulation
    temp = positions.T
    temp1 = temp[0].copy()
    temp[0] = temp[1].copy()
    temp[1] = temp1.copy()
    temp1 = temp[4].copy()
    temp[4] = temp[5].copy()
    temp[5] = temp1.copy()
    temp1 = temp[8].copy()
    temp[8] = temp[9].copy()
    temp[9] = temp1.copy()
    
    temp[12] = 1 - temp[12]
    temp[13] = 1 - temp[13]
    temp[8] = 1 - temp[8]
    temp[4] = 1 - temp[4]
    temp[0] = 1 - temp[0]
    positions = temp.T.copy()
    
    positions = positions * (posrange.T[1] - posrange.T[0]) + posrange.T[0]

    data = []
    data_len = 3000000
    last = np.zeros(16).astype(int)
    broken = False
    #go through 3000 positions
    for j in range(3000):
        maxi = 30
        posn = last.copy()
        for i in range(16):
            #first 5 positions go to P_0, otherwise go to specified position in list
            posn[i] = positions[j+3000][i] #this is from position 3000 to 6000
            if (j >= 5): maxi = max(maxi, abs(last[i] - posn[i]))
        if j < 5: posn = firstpos
        
        if broken == True: break
        num = int(maxi / 30)
        print(j)
        #break into intermediate positions from last, move
        for k in range(1, num + 1):
            pos = np.zeros(16).astype(int)
            for i in range(16):
                pos[i] = last[i] + (posn[i] - last[i]) * k / num

            #move: write pos
            failed = True
            while failed == True:
                failed = False
                for i in range(16):
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_GOAL_POSITION, pos[i])
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
                        failed = True
                    elif dxl_error != 0:
                        print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
                        broken = True
                        break
                if broken == True: break
                if failed == True:
                    time.sleep(300)
            if broken == True: break
            
            b = False
            actpos = pos.copy()
            count = 0
            while b == False:
                count += 1
                b = True
                for i in range(16):
                    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID[i], ADDR_PRESENT_POSITION)
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
                    elif dxl_error != 0:
                        print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
                        broken = True
                        break
                    actpos[i] = dxl_present_position
                    if abs(pos[i] - dxl_present_position) > THRESHOLD:
                        # print(i)
                        b = False
                if broken == True: break
                if count > 100: broken = True 

            if broken == True: break
            # get sensor data and save
            data1 = []
            sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
            sensor_data = sensor_data - baseline
            for x in range(60):
                data1.append(sensor_data[x])
            for x in range(16): data1.append(actpos[x])
            data.append(np.array(data1))
            
        if j % 30 == 0:
            data1 = np.array(data)
            np.savetxt(f"{filename}.txt", data1)  
        last = posn.copy()

    sensor_stream.pause_streaming()
    sensor_stream.join()
    data = np.array(data)
    np.savetxt(f"{filename}.txt", data)


    for i in range(16):
        packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort() 