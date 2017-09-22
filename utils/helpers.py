import time
import os
import numpy as np


def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def create_project_tree(folders_list):
    for folder in folders_list:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def get_list_from_file(filename):
    items = []
    with open(filename) as f:
        for item in f.read().splitlines():
            items.append(item)
    return items


def bounding_box(arr, padding=0, square=False):
    a = np.where(arr != 0)
    if a[0].size and a[1].size:
        min_ax0 = np.min(a[0])
        max_ax0 = np.max(a[0])
        min_ax1 = np.min(a[1])
        max_ax1 = np.max(a[1])
        if square:
            min_ax = min(min_ax0, min_ax1)
            max_ax = max(max_ax0, max_ax1)
            return min_ax - padding, max_ax + padding, min_ax - padding, max_ax + padding
        return min_ax0 - padding, max_ax0 + padding, min_ax1 - padding, max_ax1 + padding
