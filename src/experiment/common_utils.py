import os
from enum import Enum

import numpy as np


class Distribution(Enum):
    UNIFORM = 0
    NORMAL = 1
    NYCT = 2
    NYCT_SORTED = 3
    UNIFORM_10W = 4
    NORMAL_10W = 5
    NYCT_10W = 6
    NYCT_SORTED_10W = 7


data_path = {
    Distribution.UNIFORM: "../../data/table/uniform_10000w.npy",
    Distribution.NORMAL: "../../data/table/normal_10000w.npy",
    Distribution.NYCT: "../../data/table/trip_data_1_filter.npy",
    Distribution.NYCT_SORTED: "../../data/index/trip_data_1_filter_sorted.npy",
    Distribution.UNIFORM_10W: "../../data/table/uniform_10w.npy",
    Distribution.NORMAL_10W: "../../data/table/normal_10w.npy",
    Distribution.NYCT_10W: "../../data/table/trip_data_1_filter_10w.npy",
    Distribution.NYCT_SORTED_10W: "../../data/index/trip_data_1_filter_10w_sorted.npy",
}


def load_data(distribution):
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if distribution in [Distribution.NYCT, Distribution.NYCT_10W]:
        return np.load(data_path[distribution], allow_pickle=True)[:, [10, 11, -1]]
    else:
        return np.load(data_path[distribution], allow_pickle=True)
