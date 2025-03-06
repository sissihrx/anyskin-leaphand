#!/usr/bin/env python

import time
import numpy as np
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import sys
import pygame
from datetime import datetime
from anyskin import AnySkinProcess
import argparse

def visualize(port, file=None, viz_mode="3axis", scaling=7.0, record=False):
    if file is None:
        sensor_stream = AnySkinProcess(
            num_mags=20,
            port=port,
        )
        # Start sensor stream
        sensor_stream.start()
        time.sleep(2.0)
        filename = "data/data_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        load_data = np.loadtxt(file)

    pygame.init()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    bg_image_path = os.path.join(dir_path, "viz_bg.png")
    bg_image = pygame.image.load(bg_image_path)

    image_width, image_height = bg_image.get_size()
    aspect_ratio = image_height / image_width
    desired_width = 300  # Width for each board's visualization
    desired_height = int(desired_width * aspect_ratio)
    bg_image = pygame.transform.scale(bg_image, (desired_width, desired_height))
    total_width = desired_width * 2
    total_height = desired_height * 2
    window = pygame.display.set_mode((total_width, total_height), pygame.SRCALPHA)
    chip_locations_template = np.array(
        [
            [156, 168],  # center
            [99, 169],  # left
            [207, 169],  # right
            [153, 119],  # up
            [152, 222],  # down
        ]
    )


    chip_locations = np.vstack([
        chip_locations_template,
        chip_locations_template + np.array([desired_width, 0]),
        chip_locations_template + np.array([0, desired_height]),
        chip_locations_template + np.array([desired_width, desired_height]),
    ])

    chip_xy_rotations = np.tile(
        np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0]), 4
    )

    def visualize_data(data):
        data = data.reshape(-1, 3)
        data_mag = np.linalg.norm(data, axis=1)
        # Draw the chip locations
        for magid, chip_location in enumerate(chip_locations):
            if viz_mode == "magnitude":
                pygame.draw.circle(
                    window, (255, 83, 72), chip_location, data_mag[magid] / scaling
                )
            elif viz_mode == "3axis":
                if data[magid, -1] < 0:
                    width = 2
                else:
                    width = 0
                pygame.draw.circle(
                    window,
                    (255, 0, 0),
                    chip_location,
                    np.abs(data[magid, -1]) / scaling,
                    width,
                )
                arrow_start = chip_location
                rotation_mat = np.array(
                    [
                        [
                            np.cos(chip_xy_rotations[magid]),
                            -np.sin(chip_xy_rotations[magid]),
                        ],
                        [
                            np.sin(chip_xy_rotations[magid]),
                            np.cos(chip_xy_rotations[magid]),
                        ],
                    ]
                )
                data_xy = np.dot(rotation_mat, data[magid, :2])
                arrow_end = (
                    chip_location[0] + data_xy[0] / scaling,
                    chip_location[1] + data_xy[1] / scaling,
                )
                pygame.draw.line(window, (0, 255, 0), arrow_start, arrow_end, 2)

    def get_baseline():
        baseline_data = sensor_stream.get_data(num_samples=5)
        baseline_data = np.array(baseline_data)[:, 1:]
        baseline = np.mean(baseline_data, axis=0)
        return baseline

    time.sleep(1)
    if file is None:
        baseline = get_baseline()
    frame_num = 0
    running = True
    data = []
    data_len = 30000
    clock = pygame.time.Clock()
    FPS = 60
    while running:
        window.fill((230, 230, 220, 255))

        for i in range(2):
            for j in range(2):
                window.blit(bg_image, (i * desired_width, j * desired_height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                print(f"Mouse clicked at ({x}, {y})")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    # Recalculate baseline
                    baseline_data = sensor_stream.get_data(num_samples=5)
                    baseline_data = np.array(baseline_data)[:, 1:]
                    baseline = np.mean(baseline_data, axis=0)

        if file is not None:
            sensor_data = load_data[data_len]
            data_len += 24
            baseline = np.zeros_like(sensor_data)
        else:
            sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
            data.append(sensor_data - baseline)

        # Visualize data on top of the cleared layer and backgrounds
        visualize_data(sensor_data - baseline)
        pygame.display.update()
        clock.tick(FPS)
    pygame.quit()
    if file is None:
        sensor_stream.pause_streaming()
        sensor_stream.join()
        data = np.array(data)
        if record:
            np.savetxt(f"{filename}.txt", data)

def default_viz(argv=sys.argv):
    visualize(port=argv[1])

if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Test code to run a AnySkin streaming process in the background. Allows data to be collected without code blocking")
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", default="/dev/cu.usbmodem11401")
    parser.add_argument("-f", "--file", type=str, help="path to load data from", default=None)
    parser.add_argument("-v", "--viz_mode", type=str, help="visualization mode", default="3axis", choices=["magnitude", "3axis"])
    parser.add_argument("-s", "--scaling", type=float, help="scaling factor for visualization", default=7.0)
    parser.add_argument('-r', '--record', action='store_true', help='record data')
    args = parser.parse_args()
    # fmt: on
    visualize(args.port, args.file, args.viz_mode, args.scaling, args.record)