import csv
import os

import numpy as np
import pandas

from src.utils.common_utils import Point
from src.utils.geohash_utils import Geohash


class MyError(Exception):
    def __init__(self, message):
        self.message = message


def csv_to_npy(input_path, output_path):
    data_list = pandas.read_csv(input_path).values[:, [10, 11]]
    # 时间字符串转时间戳
    t_list = pandas.read_csv(input_path).values[:, [5]].astype(np.datetime64).astype(np.int32).reshape(
        data_list.shape[0])
    data_list = np.insert(data_list, 2, t_list, axis=1)
    # 根据时间戳排序
    data_list = data_list[t_list.argsort()]
    np.save(output_path, data_list)


def count_csv(path):
    with open(path) as f1:
        reader = csv.reader(f1)
        count = 0
        for line in reader:
            count += 1
        return count


def filter_row_in_region(input_path, output_path, range_limit):
    data_list = np.load(input_path, allow_pickle=True)
    contain_filter = np.apply_along_axis(lambda x: range_limit.contain(Point(x[0], x[1])), axis=1, arr=data_list)
    contain_data_list = data_list[contain_filter]
    np.save(output_path, contain_data_list)
    print("Data num in region: %d" % contain_data_list.shape[0])
    mins = np.min(contain_data_list, axis=0)
    maxs = np.max(contain_data_list, axis=0)
    print("Space scope: %f, %f, %f, %f" % (mins[1], maxs[1], mins[0], maxs[0]))
    print("Time scope: %d, %d" % (mins[2], maxs[2]))


def add_key_field(input_path, output_path, first_key):
    data_list = np.load(input_path, allow_pickle=True).astype(object)
    data_len = len(data_list)
    key_list = np.arange(first_key, first_key + data_len)
    key_list.resize((data_len, 1))
    data_list = np.hstack((data_list, key_list))
    np.save(output_path, data_list)


def sample(input_path, output_path, lines_limit):
    data_list = np.load(input_path, allow_pickle=True)
    np.random.seed(1)
    sample_key = np.random.randint(0, len(data_list) - 1, size=lines_limit)
    sample_key = sample_key[sample_key.argsort()]
    np.save(output_path, data_list[sample_key])


def create_point_query(input_path, output_path, query_number_limit):
    data_list = np.load(input_path, allow_pickle=True)
    np.random.seed(1)
    sample_key = np.random.randint(0, len(data_list) - 1, size=query_number_limit)
    sample_data_list = data_list[sample_key]
    sample_data_list = np.array([[data[0], data[1]] for data in sample_data_list])
    np.save(output_path, sample_data_list)


def create_range_query(output_path, data_range, query_number_limit, range_ratio_list):
    result = np.empty(shape=(0, 5))
    data_range_width = data_range[3] - data_range[2]
    data_range_height = data_range[1] - data_range[0]
    for range_ratio in range_ratio_list:
        range_dim_ratio = range_ratio ** 0.5
        child_range_width = data_range_width * range_dim_ratio
        child_range_height = data_range_height * range_dim_ratio
        child_range_list = []
        while len(child_range_list) < query_number_limit:
            point_x1_list = np.random.randint(0, 100000, size=query_number_limit) / 100000 * data_range_width + \
                            data_range[2]
            point_y1_list = np.random.randint(0, 100000, size=query_number_limit) / 100000 * data_range_height + \
                            data_range[0]
            child_range_list.extend([[point_y1_list[i], point_y1_list[i] + child_range_height,
                                      point_x1_list[i], point_x1_list[i] + child_range_width]
                                     for i in range(query_number_limit)
                                     if point_y1_list[i] + child_range_height < data_range[1]
                                     and point_x1_list[i] + child_range_width < data_range[3]])
        range_ratio_list = np.array([[range_ratio]] * query_number_limit)
        child_ranges = np.hstack((np.array(child_range_list[:query_number_limit]), range_ratio_list))
        result = np.vstack((result, child_ranges))
    np.save(output_path, result)


def create_knn_query(input_path, output_path, query_number_limit, n_list):
    data_list = np.load(input_path, allow_pickle=True)
    data_len = len(data_list)
    result = np.empty(shape=(0, 3))
    for n in n_list:
        np.random.seed(n)
        sample_key = np.random.randint(0, data_len - 1, size=query_number_limit)
        n_data_list = data_list[sample_key]
        n_data_list = np.array([[data[0], data[1]] for data in n_data_list])
        n_list = np.array([[n]] * query_number_limit)
        n_data_list = np.hstack((n_data_list, n_list))
        result = np.vstack((result, n_data_list))
    np.save(output_path, result)


def geohash_and_sort(input_path, output_path, data_precision, region):
    geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
    data_list = np.load(input_path, allow_pickle=True)
    data_list = [(data[0], data[1], geohash.encode(data[0], data[1]), data[2], data[3]) for data in data_list]
    data_list.sort(key=lambda x: x[2])
    np.save(output_path, data_list)


def npy_to_table(input_path, output_path, is_sorted):
    data_list = np.load(input_path, allow_pickle=True)
    if is_sorted:
        dtype = [("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]
    else:
        dtype = [("0", 'f8'), ("1", 'f8'), ("2", 'i4'), ("3", 'i4')]
    np.save(output_path, np.array([tuple(data) for data in data_list.tolist()], dtype=dtype))


def synthetic_data(output_path, data_size, spatial_scope, time_scope, data_precision, type):
    # time is sorted and uniform
    t = np.arange(time_scope[0], time_scope[1] + 1, (time_scope[1] + 1 - time_scope[0]) / data_size).astype(np.int32)
    if type == 'uniform':
        x = np.around(np.random.uniform(spatial_scope[2], spatial_scope[3], size=data_size), decimals=data_precision)
        y = np.around(np.random.uniform(spatial_scope[0], spatial_scope[1], size=data_size), decimals=data_precision)
    elif type == 'normal':
        x_redius = (spatial_scope[3] - spatial_scope[2]) / 2
        y_redius = (spatial_scope[1] - spatial_scope[0]) / 2
        x_center = (spatial_scope[3] + spatial_scope[2]) / 2
        y_center = (spatial_scope[1] + spatial_scope[0]) / 2
        x = np.random.normal(0, 1, size=data_size)
        y = np.random.normal(0, 1, size=data_size)
        x = np.around(x * x_redius / max(-x.min(), x.max()) + x_center, decimals=data_precision)
        y = np.around(y * y_redius / max(-y.min(), y.max()) + y_center, decimals=data_precision)
        # 单独处理1：1作为最大值，在geohash编码时，会超出长度限制，比如8位小数，0-1范围，geohash编码为1000...000，长度31，超出30限制
        x[x == 1] = 1 - pow(10, -data_precision)
        y[y == 1] = 1 - pow(10, -data_precision)
    else:
        return
    np.save(output_path, np.stack((x, y, t), axis=1))


def create_distinct_data(input_path, output_path):
    data = np.load(input_path)
    distinct_data = np.unique(data, axis=0)
    np.save(output_path, distinct_data)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # 1. 把csv转npy
    # 数据来源：从http://www.andresmh.com/nyctaxitrips/下载trip_data.7z，拿到其中的一月份和二月份数据csv，转成npy
    # npy是外存结构，size=128Byte的头文件大小+数据大小，csv的内容都是string
    # input_path = r"D:\科研\毕业论文-新\trip_data_1.csv"
    # output_path = "./table/nyct_1.npy"
    # csv_to_npy(input_path, output_path)
    # input_path = r"D:\科研\毕业论文-新\trip_data_2.csv"
    # output_path = "./table/nyct_2.npy"
    # csv_to_npy(input_path, output_path)
    # 2. 数据清洗，只选取region内
    # 数据总记录数：14776615+13990176，1.50GB+1.39GB
    # Region(40, 42, -75, -73)内14507253+13729724=28236977，1.47GB+1.37GB=2.84GB
    # Region(40.61, 40.87, -74.05, -73.76)内：14494739+13716799，时间范围1356998400-1359676799-1362095999
    # input_path = "./table/nyct_1.npy"
    # output_path = "./table/nyct_1.npy"
    # filter_row_in_region(input_path, output_path, Region(40.61, 40.87, -74.05, -73.76))
    # input_path = "./table/nyct_2.npy"
    # output_path = "./table/nyct_2.npy"
    # filter_row_in_region(input_path, output_path, Region(40.61, 40.87, -74.05, -73.76))
    # 3. 生成uniform和normal的数据
    # 精度8，范围0-1-0-1，geohash长度为60，精度9的话长度就66了，超过了int64的范围
    # output_path = "./table/uniform_1.npy"
    # synthetic_data(output_path, 14494739, [0, 1, 0, 1], [1356998400, 1359676799], 8, 'uniform')
    # output_path = "./table/uniform_2.npy"
    # synthetic_data(output_path, 13716799, [0, 1, 0, 1], [1359676800, 1362095999], 8, 'uniform')
    # output_path = "./table/normal_1.npy"
    # synthetic_data(output_path, 14494739, [0, 1, 0, 1], [1356998400, 1359676799], 8, 'normal')
    # output_path = "./table/normal_2.npy"
    # synthetic_data(output_path, 13716799, [0, 1, 0, 1], [1359676800, 1362095999], 8, 'normal')
    # 4. 生成10w的数据
    # input_path = "./table/uniform_1.npy"
    # output_path = './table/uniform_1_10w.npy'
    # sample(input_path, output_path, 100000)
    # input_path = "./table/uniform_2.npy"
    # output_path = './table/uniform_2_10w.npy'
    # sample(input_path, output_path, 100000)
    # input_path = "./table/normal_1.npy"
    # output_path = './table/normal_1_10w.npy'
    # sample(input_path, output_path, 100000)
    # input_path = "./table/normal_2.npy"
    # output_path = './table/normal_2_10w.npy'
    # sample(input_path, output_path, 100000)
    # input_path = "./table/nyct_1.npy"
    # output_path = './table/nyct_1_10w.npy'
    # sample(input_path, output_path, 100000)
    # input_path = "./table/nyct_2.npy"
    # output_path = './table/nyct_2_10w.npy'
    # sample(input_path, output_path, 100000)
    # [Optional] 5. 生成不重复的数据
    # input_path = "./table/nyct_1_10w.npy"
    # output_path = "./table/nyct_1_10w_distinct.npy"
    # create_distinct_data(input_path, output_path)
    # 6. 生成索引列
    # output_path = "./table/uniform_1.npy"
    # output_path = "./table/uniform_1_10w.npy"
    # output_path = "./table/normal_1.npy"
    # output_path = "./table/normal_1_10w.npy"
    # output_path = "./table/nyct_1.npy"
    # output_path = "./table/nyct_1_10w.npy"
    # first_key = 0
    # output_path = "./table/uniform_2.npy"
    # output_path = "./table/normal_2.npy"
    # output_path = "./table/nyct_2.npy"
    # first_key = 14494739
    # output_path = "./table/uniform_2_10w.npy"
    # output_path = "./table/normal_2_10w.npy"
    # output_path = "./table/nyct_2_10w.npy"
    # first_key = 100000
    # add_key_field(output_path, output_path, first_key)
    # 7. Geohash排序数据
    # input_path = "./table/uniform_1.npy"
    # output_path = "./index/uniform_1_sorted.npy"
    # input_path = "./table/normal_1.npy"
    # output_path = "./index/normal_1_sorted.npy"
    # data_precision = 8
    # region = Region(0, 1, 0, 1)
    # input_path = "./table/nyct_1.npy"
    # output_path = "./index/nyct_1_sorted.npy"
    # input_path = "./table/nyct_1_10w.npy"
    # output_path = "./index/nyct_1_10w_sorted.npy"
    # data_precision = 6
    # region = Region(40.61, 40.87, -74.05, -73.76)
    # geohash_and_sort(input_path, output_path, data_precision, region)
    # 8 npy转标准table表格存储：xyzti为'f8, f8, i8, i4, i4'
    # output_path = "./table/uniform_1.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/uniform_1_10w.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/normal_1.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/normal_1_10w.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/nyct_1.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/nyct_1_10w.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/nyct_2.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./table/nyct_2_10w.npy"
    # npy_to_table(output_path, output_path, False)
    # output_path = "./index/uniform_1_sorted.npy"
    # npy_to_table(output_path, output_path, True)
    # output_path = "./index/normal_1_sorted.npy"
    # npy_to_table(output_path, output_path, True)
    # output_path = "./index/nyct_1_sorted.npy"
    # npy_to_table(output_path, output_path, True)
    # output_path = "./index/nyct_1_10w_sorted.npy"
    # npy_to_table(output_path, output_path, True)
    # 1. 生成point检索范围
    # input_path = './table/uniform_1.npy'
    # output_path = './query/point_query_uniform.npy'
    # input_path = './table/normal_1.npy'
    # output_path = './query/point_query_normal.npy'
    # input_path = './table/nyct_1.npy'
    # output_path = './query/point_query_nyct.npy'
    # query_number_limit = 1000
    # create_point_query(input_path, output_path, query_number_limit)
    # 2. 生成range检索范围
    # output_path = './query/range_query_uniform.npy'
    # data_range = [0, 1, 0, 1]
    # output_path = './query/range_query_normal.npy'
    # data_range = [0, 1, 0, 1]
    # range_ratio_list = [0.000006, 0.000025, 0.0001, 0.0004, 0.0016]
    # output_path = './query/range_query_nyct.npy'
    # data_range = [40.61, 40.87, -74.05, -73.76]
    # query_number_limit = 1000
    # create_range_query(output_path, data_range, query_number_limit, range_ratio_list)
    # 3.生成knn检索范围
    # input_path = './table/uniform_1.npy'
    # output_path = './query/knn_query_uniform.npy'
    # input_path = './table/normal_1.npy'
    # output_path = './query/knn_query_normal.npy'
    # input_path = './table/nyct_1.npy'
    # output_path = './query/knn_query_nyct.npy'
    # query_number_limit = 1000
    # n_list = [4, 8, 16, 32, 64]
    # create_knn_query(input_path, output_path, query_number_limit, n_list)
    paths = ['./table/uniform_2_10w.npy', './table/uniform_2.npy',
             './table/normal_2_10w.npy', './table/normal_2.npy',
             './table/nyct_2_10w.npy', './table/nyct_2.npy']
    end_time = 1362096000
    for path in paths:
        data_list = np.load(path, allow_pickle=True)
        i = -1
        while data_list[i][-2] >= end_time:
            i -= 1
        if i != -1:
            data_list = data_list[:i + 1]
            np.save(path, data_list)
