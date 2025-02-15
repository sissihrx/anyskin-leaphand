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
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", default="/dev/cu.usbmodem11401")
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
    # start sensor stream
    sensor_stream.start()
    time.sleep(1.0)
    filename = "fullmodeldata/fullrandcont"
    pygame.init()
    time.sleep(0.1)
    baseline = get_baseline()

    # addresses
    ADDR_PROFILE_VELOCITY = 112 
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    DXL_ID = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    BAUDRATE = 3000000
    PORT = '/dev/tty.usbserial-FT7WBF78'  # port 
    THRESHOLD = 50

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
    

    data = []
    data_len = 3000000
    last = np.zeros(16).astype(int)
    for j in range(150):
        maxi = 20
        posn = np.zeros(16).astype(int)
        for i in range(16):
            posn[i] = random.randint(posrange[i][0], posrange[i][1])
            if (j > 0): maxi = max(maxi, abs(last[i] - posn[i]))
    
        num = int(maxi / 20)
        for k in range(1, num + 1):
            pos = np.zeros(16).astype(int)
            for i in range(16):
                pos[i] = last[i] + (posn[i] - last[i]) * k / num

            #move: write pos
            for i in range(16):
                dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID[i], ADDR_GOAL_POSITION, pos[i])
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
                elif dxl_error != 0:
                    print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
            
            b = False
            while b == False:
                b = True
                for i in range(16):
                    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID[i], ADDR_PRESENT_POSITION)
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Communication failed: {packetHandler.getTxRxResult(dxl_comm_result)}")
                    elif dxl_error != 0:
                        print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
                    if abs(pos[i] - dxl_present_position) > THRESHOLD:
                        b = False

            # get sensor data
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        # Recalculate baseline
                        baseline_data = sensor_stream.get_data(num_samples=5)
                        baseline_data = np.array(baseline_data)[:, 1:]
                        baseline = np.mean(baseline_data, axis=0)

            data1 = []
            if file is not None:
                load_data = np.loadtxt(file)
                sensor_data = load_data[data_len]
                data_len += 24
                baseline = np.zeros_like(sensor_data)
                for x in sensor_data - baseline:
                    data1.append(x)
                for i in range(16):
                    data1.append(pos[i])
            else:
                sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
                # print(sensor_data)
                for x in sensor_data - baseline:
                    data1.append(x)
                for i in range(16):
                    data1.append(pos[i])
                data.append(np.array(data1))
        for i in range(16): 
            last[i] = posn[i]

    if file is None:
        sensor_stream.pause_streaming()
        sensor_stream.join()
        data = np.array(data)
        np.savetxt(f"{filename}.txt", data)

    # print(data.shape)

    for i in range(16):
        packetHandler.write1ByteTxRx(portHandler, DXL_ID[i], ADDR_TORQUE_ENABLE, 0)

    portHandler.closePort() 