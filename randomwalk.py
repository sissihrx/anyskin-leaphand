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
    filename = "fullmodeldata/randwalkdata"
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

    posrange = np.array([[1918, 1918], #here
                 [2400, 2900], #2000
                 [1800, 2600],
                 [2900, 3900],
                 [2058, 2058], #here
                 [2400, 2900], #
                 [1800, 2600],
                 [2900, 3900],
                 [2155, 2155], #here
                 [2400, 3100],
                 [1900, 2400],
                 [2000, 3000],
                 [1884, 1884], #here
                 [2100, 4000],
                 [2100, 2100], #here
                 [2200, 3100]
        ])
    firstpos = np.array([1918, 2662, 2427, 3886, 2058, 2595, 2010, 3600, 2155, 2536, 2250, 2444, 1884, 3466, 2100, 2594])

    data = []
    data_len = 3000000
    for j in range(5000):
        posn = np.zeros(16).astype(int)
        steps = 20
        for i in range(16):
            posn[i] = random.randint(posrange[i][0], posrange[i][1])
        if j < 5: 
            posn = firstpos.copy()
            steps = 1
        
        print(j)
        for k in range(steps):
            print(posn)
            #move: write pos
            failed = True
            while failed == True:
                failed = False
                for i in range(16):
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_GOAL_POSITION, posn[i])
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
                        failed = True
                    elif dxl_error != 0:
                        print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
                if failed == True:
                    time.sleep(300)

            b = False
            actpos = posn.copy()
            while b == False:
                b = True
                for i in range(16):
                    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID[i], ADDR_PRESENT_POSITION)
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
                    elif dxl_error != 0:
                        print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
                    actpos[i] = dxl_present_position
                    if abs(posn[i] - dxl_present_position) > THRESHOLD:
                        b = False

            # get sensor data
            data1 = []
            sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
            sensor_data = sensor_data - baseline
            for x in range(60):
                data1.append(sensor_data[x])
            for x in range(16): data1.append(actpos[x])
            data.append(np.array(data1))

            for i in range(16):
                dir = np.array([-1, 1])
                if (posn[i] + 30 > posrange[i][1]): dir[1] = 0
                if (posn[i] - 30 < posrange[i][0]): dir[0] = 0
                posn[i] += dir[random.randint(0, 1)] * 30
            
        if j % 2 == 0:
            data1 = np.array(data)
            np.savetxt(f"{filename}.txt", data1)  

    print("moved")
    sensor_stream.pause_streaming()
    sensor_stream.join()
    data = np.array(data)
    np.savetxt(f"{filename}.txt", data)


    for i in range(16):
        packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort() 