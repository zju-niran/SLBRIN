import os
import shutil
import time
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


# data for build index
build_data_path = {
    Distribution.UNIFORM: "../../data/table/uniform_1.npy",
    Distribution.NORMAL: "../../data/table/normal_1.npy",
    Distribution.NYCT: "../../data/table/nyct_1.npy",
    Distribution.UNIFORM_10W: "../../data/table/uniform_1_10w.npy",
    Distribution.NORMAL_10W: "../../data/table/normal_1_10w.npy",
    Distribution.NYCT_10W: "../../data/table/nyct_1_10w.npy",
    Distribution.UNIFORM_SORTED: "../../data/index/uniform_1_sorted.npy",
    Distribution.NORMAL_SORTED: "../../data/index/normal_1_sorted.npy",
    Distribution.NYCT_SORTED: "../../data/index/nyct_1_sorted.npy",
    Distribution.NYCT_10W_SORTED: "../../data/index/nyct_1_10w_sorted.npy",
}
# data for update index
update_data_path = {
    Distribution.UNIFORM: "../../data/table/uniform_2.npy",
    Distribution.NORMAL: "../../data/table/normal_2.npy",
    Distribution.NYCT: "../../data/table/nyct_2.npy",
    Distribution.UNIFORM_10W: "../../data/table/uniform_2_10w.npy",
    Distribution.NORMAL_10W: "../../data/table/normal_2_10w.npy",
    Distribution.NYCT_10W: "../../data/table/nyct_2_10w.npy",
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


def load_data(distribution, type):
    if type == 0:
        return np.load(build_data_path[distribution], allow_pickle=True)
    else:
        return np.load(update_data_path[distribution], allow_pickle=True)


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
    if type == 0:
        query_path = point_query_path[distribution]
    elif type == 1:
        query_path = range_query_path[distribution]
    else:
        query_path = knn_query_path[distribution]
    return np.load(query_path, allow_pickle=True)


def copy_dirs(from_file, to_file, ignore_file=None):
    if not os.path.exists(to_file):  # 如不存在目标目录则创建
        os.makedirs(to_file)
    files = os.listdir(from_file)  # 获取文件夹中文件和目录列表
    for f in files:
        if f == ignore_file:
            continue
        if os.path.isdir(from_file + '/' + f):  # 判断是否是文件夹
            copy_dirs(from_file + '/' + f, to_file + '/' + f)  # 递归调用本函数
        else:
            shutil.copy(from_file + '/' + f, to_file + '/' + f)  # 拷贝文件

def test_point_query(index, data_distribution):
    sum_search_time = 0
    sum_io_cost = 0
    io_cost = index.io_cost
    point_query_list = load_query(data_distribution, 0).tolist()
    for k in range(5):
        start_time = time.time()
        index.test_point_query(point_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(point_query_list)
        sum_search_time += search_time
        sum_io_cost += (index.io_cost - io_cost) / len(point_query_list)
        io_cost = index.io_cost
    return sum_search_time / 5, sum_io_cost / 5


def test_query(index, data_distribution, type):
    index_test_querys = [index.test_point_query, index.test_range_query, index.test_knn_query]
    index_test_query = index_test_querys[type]
    sum_search_time = 0
    sum_io_cost = 0
    io_cost = index.io_cost
    query_list = load_query(data_distribution, type).tolist()
    query_list_len = len(query_list)
    # 查询跑多次，减小算力波动的影响
    for k in range(5):
        start_time = time.time()
        index_test_query(query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / query_list_len
        sum_search_time += search_time
        sum_io_cost += (index.io_cost - io_cost) / query_list_len
        io_cost = index.io_cost
    return sum_search_time / 5, sum_io_cost / 5
