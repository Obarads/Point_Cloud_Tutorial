import time
import numpy as np
import random
import os

def env_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def single_color(color, num_points: int):
    """
    Args:
        color: RGB (3) or color code
        num_points: number of points
    Return:
        color poinsts: (num_points, 3)
    """
    if type(color) == str:
        color = np.array(
            [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
        )
    return np.tile([color], (num_points, 1))


def color_range_rgb_to_8bit_rgb(colors: np.ndarray, color_range: list = [0, 1]):
    # Get color range of minimum and maximum
    min_color = color_range[0]
    max_color = color_range[1]

    # color range (min_color ~ max_color) to (0 ~ max_color-min_color)
    colors -= min_color
    max_color -= min_color

    # to 0 ~ 255 color range and uint32 type
    colors = colors / max_color * 255
    colors = colors.astype(np.uint32)

    return colors


def rgb_to_hex(rgb):
    hex = (rgb[:, 0] << 16) + (rgb[:, 1] << 8) + rgb[:, 2]
    return hex


def time_watcher(previous_time=None, print_key=""):
    current_time = time.time()
    if previous_time is None:
        print("time_watcher start")
    else:
        print("{}: {}".format(print_key, current_time - previous_time))
    return current_time
