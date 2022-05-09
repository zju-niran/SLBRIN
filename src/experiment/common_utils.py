from enum import Enum

import numpy as np

from src.spatial_index.common_utils import Region


class Distribution(Enum):
    UNIFORM = 0
    NORMAL = 1
    NYCT = 2
    UNIFORM_10W = 3
    NORMAL_10W = 4
    NYCT_10W = 5
    UNIFORM_SORTED = 6
    NORMAL_SORTED = 7
    NYCT_SORTED = 8
    NYCT_10W_SORTED = 9


data_path = {
    Distribution.UNIFORM: "../../data/table/uniform_10000w.npy",
    Distribution.NORMAL: "../../data/table/normal_10000w.npy",
    Distribution.NYCT: "../../data/table/trip_data_1_filter.npy",
    Distribution.UNIFORM_10W: "../../data/table/uniform_10w.npy",
    Distribution.NORMAL_10W: "../../data/table/normal_10w.npy",
    Distribution.NYCT_10W: "../../data/table/trip_data_1_filter_10w.npy",
    Distribution.UNIFORM_SORTED: "../../data/index/uniform_sorted.npy",
    Distribution.NORMAL_SORTED: "../../data/index/normal_sorted.npy",
    Distribution.NYCT_SORTED: "../../data/index/nyct_sorted.npy",
    Distribution.NYCT_10W_SORTED: "../../data/index/nyct_10w_sorted.npy",
}

data_precision = {
    Distribution.UNIFORM: 8,
    Distribution.NORMAL: 8,
    Distribution.NYCT: 6,
    Distribution.UNIFORM_10W: 8,
    Distribution.NORMAL_10W: 8,
    Distribution.NYCT_10W: 6,
    Distribution.UNIFORM_SORTED: 8,
    Distribution.NORMAL_SORTED: 8,
    Distribution.NYCT_SORTED: 6,
    Distribution.NYCT_10W_SORTED: 6,
}

data_region = {
    Distribution.UNIFORM: Region(0, 1, 0, 1),
    Distribution.NORMAL: Region(0, 1, 0, 1),
    Distribution.NYCT: Region(40, 42, -75, -73),
    Distribution.UNIFORM_10W: Region(0, 1, 0, 1),
    Distribution.NORMAL_10W: Region(0, 1, 0, 1),
    Distribution.NYCT_10W: Region(40, 42, -75, -73),
    Distribution.UNIFORM_SORTED: Region(0, 1, 0, 1),
    Distribution.NORMAL_SORTED: Region(0, 1, 0, 1),
    Distribution.NYCT_SORTED: Region(40, 42, -75, -73),
    Distribution.NYCT_10W_SORTED: Region(40, 42, -75, -73),
}


def load_data(distribution):
    if distribution in [Distribution.NYCT, Distribution.NYCT_10W]:
        return np.load(data_path[distribution], allow_pickle=True)[:, [10, 11, -1]]
    else:
        return np.load(data_path[distribution], allow_pickle=True)


point_query_path = {
    Distribution.UNIFORM: '../../data/query/point_query_uniform.npy',
    Distribution.NORMAL: '../../data/query/point_query_normal.npy',
    Distribution.NYCT: '../../data/query/point_query_nyct.npy',
    Distribution.UNIFORM_10W: '../../data/query/point_query_uniform.npy',
    Distribution.NORMAL_10W: '../../data/query/point_query_normal.npy',
    Distribution.NYCT_10W: '../../data/query/point_query_nyct.npy',
    Distribution.UNIFORM_SORTED: '../../data/query/point_query_uniform.npy',
    Distribution.NORMAL_SORTED: '../../data/query/point_query_normal.npy',
    Distribution.NYCT_SORTED: '../../data/query/point_query_nyct.npy',
    Distribution.NYCT_10W_SORTED: '../../data/query/point_query_nyct.npy',
}

range_query_path = {
    Distribution.UNIFORM: '../../data/query/range_query_uniform.npy',
    Distribution.NORMAL: '../../data/query/range_query_normal.npy',
    Distribution.NYCT: '../../data/query/range_query_nyct.npy',
    Distribution.UNIFORM_10W: '../../data/query/range_query_uniform.npy',
    Distribution.NORMAL_10W: '../../data/query/range_query_normal.npy',
    Distribution.NYCT_10W: '../../data/query/range_query_nyct.npy',
    Distribution.UNIFORM_SORTED: '../../data/query/range_query_uniform.npy',
    Distribution.NORMAL_SORTED: '../../data/query/range_query_normal.npy',
    Distribution.NYCT_SORTED: '../../data/query/range_query_nyct.npy',
    Distribution.NYCT_10W_SORTED: '../../data/query/range_query_nyct.npy',
}

knn_query_path = {
    Distribution.UNIFORM: '../../data/query/knn_query_uniform.npy',
    Distribution.NORMAL: '../../data/query/knn_query_normal.npy',
    Distribution.NYCT: '../../data/query/knn_query_nyct.npy',
    Distribution.UNIFORM_10W: '../../data/query/knn_query_uniform.npy',
    Distribution.NORMAL_10W: '../../data/query/knn_query_normal.npy',
    Distribution.NYCT_10W: '../../data/query/knn_query_nyct.npy',
    Distribution.UNIFORM_SORTED: '../../data/query/knn_query_uniform.npy',
    Distribution.NORMAL_SORTED: '../../data/query/knn_query_normal.npy',
    Distribution.NYCT_SORTED: '../../data/query/knn_query_nyct.npy',
    Distribution.NYCT_10W_SORTED: '../../data/query/knn_query_nyct.npy',
}


def load_query(distribution, type):
    if type == "point":
        query_path = point_query_path[distribution]
    elif type == "range":
        query_path = range_query_path[distribution]
    else:
        query_path = knn_query_path[distribution]
    return np.load(query_path, allow_pickle=True)
