import csv
import os

import numpy as np
import pandas

from src.spatial_index.common_utils import Point
from src.spatial_index.geohash_utils import Geohash


class MyError(Exception):

    def __init__(self, message):
        self.message = message


def csv_to_npy(input_path, output_path):
    np.save(output_path, pandas.read_csv(input_path).values)


def count_csv(path):
    with open(path) as f1:
        reader = csv.reader(f1)
        count = 0
        for line in reader:
            count += 1
        return count


def filter_row_in_region(input_path, output_path, range_limit):
    data_list = np.load(input_path, allow_pickle=True)
    contain_filter = np.apply_along_axis(lambda x: range_limit.contain(Point(x[10], x[11])), axis=1, arr=data_list)
    contain_data_list = data_list[contain_filter]
    np.save(output_path, contain_data_list)


def add_key_field(input_path, output_path, first_key):
    data_list = np.load(input_path, allow_pickle=True).astype(np.object)
    data_len = len(data_list)
    key_list = np.arange(first_key, first_key + data_len)
    key_list.resize((data_len, 1))
    data_list = np.hstack((data_list, key_list))
    np.save(output_path, data_list)


def get_region(input_path):
    df = pandas.read_csv(input_path)
    print("Region: %f, %f, %f, %f" % (df.y.min(), df.y.max(), df.x.min(), df.x.max()))


def sample(input_path, output_path, lines_limit):
    data_list = np.load(input_path, allow_pickle=True)
    np.random.seed(1)
    sample_key = np.random.randint(0, len(data_list) - 1, size=lines_limit)
    np.save(output_path, data_list[sample_key])


def create_point_query(input_path, output_path, query_number_limit):
    if "nyct" in output_path:
        data_list = np.load(input_path, allow_pickle=True)[:, [10, 11]]
    else:
        data_list = np.load(input_path, allow_pickle=True)[:, [0, 1]]
    np.random.seed(1)
    sample_key = np.random.randint(0, len(data_list) - 1, size=query_number_limit)
    np.save(output_path, data_list[sample_key])


def create_range_query(output_path, data_range, query_number_limit, range_ratio_list):
    result = np.empty(shape=(0, 5))
    data_range_width = data_range[3] - data_range[2]
    data_range_height = data_range[1] - data_range[0]
    for range_ratio in range_ratio_list:
        child_range_width = data_range_width * range_ratio
        child_range_height = data_range_height * range_ratio
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
    if "nyct" in output_path:
        data_list = np.load(input_path, allow_pickle=True)[:, [10, 11]]
    else:
        data_list = np.load(input_path, allow_pickle=True)[:, [0, 1]]
    data_len = len(data_list)
    result = np.empty(shape=(0, 3))
    for n in n_list:
        np.random.seed(n)
        sample_key = np.random.randint(0, data_len - 1, size=query_number_limit)
        n_data_list = data_list[sample_key]
        n_list = np.array([[n]] * query_number_limit)
        n_data_list = np.hstack((n_data_list, n_list))
        result = np.vstack((result, n_data_list))
    np.save(output_path, result)


def geohash_and_sort(input_path, output_path, data_precision, region):
    geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
    if "nyct" in output_path:
        data = np.load(input_path, allow_pickle=True)[:, [10, 11, -1]]
    else:
        data = np.load(input_path, allow_pickle=True)
        data = [(data[i][0], data[i][1], geohash.encode(data[i][0], data[i][1]), data[i][2]) for i in range(len(data))]
    data = sorted(data, key=lambda x: x[2])
    np.save(output_path, np.array(data, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4')]))


def create_data(output_path, data_size, scope, data_precision, type):
    if type == 'uniform':
        x = np.around(np.random.uniform(scope[2], scope[3], size=data_size), decimals=data_precision)
        y = np.around(np.random.uniform(scope[0], scope[1], size=data_size), decimals=data_precision)
    elif type == 'normal':
        x_redius = (scope[3] - scope[2]) / 2
        y_redius = (scope[1] - scope[0]) / 2
        x_center = (scope[3] + scope[2]) / 2
        y_center = (scope[1] + scope[0]) / 2
        x = np.random.normal(0, 1, size=data_size)
        y = np.random.normal(0, 1, size=data_size)
        x = np.around(x * x_redius / max(-x.min(), x.max()) + x_center, decimals=data_precision)
        y = np.around(y * y_redius / max(-y.min(), y.max()) + y_center, decimals=data_precision)
        # 单独处理1：1作为最大值，在geohash编码时，会超出长度限制，比如8位小数，0-1范围，geohash编码为1000...000，长度31，超出30限制
        x[x == 1] = 1 - pow(10, -data_precision)
        y[y == 1] = 1 - pow(10, -data_precision)
    else:
        return
    np.save(output_path, np.stack((x, y), axis=1))


def plot_npy(input_path):
    from matplotlib import pyplot as plt
    data = np.load(input_path)
    plt.scatter(data[:, 0], data[:, 1])
    plt.legend()
    plt.show()


def create_distinct_data(input_path, output_path):
    data = np.load(input_path)
    distinct_data = np.unique(data, axis=0)
    np.save(output_path, distinct_data)


def log_to_csv(input_path, output_path):
    """
    提取指定文件夹里的非csv文件的数据，存为同名的csv文件
    """
    files = os.listdir(input_path)
    for file in files:
        result = []
        filename, suffix = file.split(".")
        if suffix == "csv":
            continue
        output_file = filename + ".csv"
        with open(os.path.join(input_path, file), 'r') as f1,\
                open(os.path.join(output_path, output_file), 'w', newline='') as f2:
            data = []
            for line in f1.readlines():
                if 'start' in line:
                    result.append(data)
                    data = []
                else:
                    value = line.split(':')[-1][:-1]
                    data.append(value)
            result.append(data)
            csv_w = csv.writer(f2)
            for data in result:
                if len(data):
                    csv_w.writerow(data)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    headers = ['medallion',
               'hack_license',
               'vendor_id',
               'rate_code',
               'store_and_fwd_flag',
               'pickup_datetime',
               'dropoff_datetime',
               'passenger_count',
               'trip_time_in_secs',
               'trip_distance',
               'pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude']

    # 1. 把csv转npy
    # 数据来源：从http://www.andresmh.com/nyctaxitrips/下载trip_data.7z，拿到其中的一月份和二月份数据csv，转成npy
    # npy是外存结构，size=128Byte的头文件大小+数据大小，csv的内容都是string
    # input_path = "./table/trip_data_1.csv"
    # output_path = "./table/trip_data_1.npy"
    # input_path = "./table/trip_data_2.csv"
    # output_path = "./table/trip_data_2.npy"
    # csv_to_npy(input_path, output_path)
    # 2. 数据清洗，只选取region内且
    # 数据总记录数：14776615/13990176，region内14507253/13729724
    # 文件size：1.50GB/1.39GB，region内1.47GB/1.37GB
    # input_path = "./table/trip_data_1.npy"
    # input_path = "./table/trip_data_2.npy"
    # output_path = "./table/trip_data_1_filter.npy"
    # output_path = "./table/trip_data_2_filter.npy"
    # filter_row_in_region(input_path, output_path, Region(40, 42, -75, -73))
    # 输出数据spatial scope
    # 40.016666, 41.933331, -74.990433, -73.000938
    # input_path = "./table/trip_data_1_filter.npy"
    # get_region(input_path)
    # 3. 生成uniform和normal的数据
    # 精度8，范围0-1-0-1，geohash长度为60，精度9的话长度就66了，超过了int64的范围
    # output_path_uniform = "./table/uniform_10000w.npy"
    # output_path_normal = "./table/normal_10000w.npy"
    # create_data(output_path_uniform, 100000000, [0, 1, 0, 1], 8, 'uniform')
    # create_data(output_path_normal, 100000000, [0, 1, 0, 1], 8, 'normal')
    # plot_npy(output_path_uniform)
    # plot_npy(output_path_normal)
    # 4. 生成10w的数据
    # input_path = "./table/uniform_10000w.npy"
    # output_path_10w_sample = './table/uniform_10w.npy'
    # input_path = "./table/normal_10000w.npy"
    # output_path_10w_sample = './table/normal_10w.npy'
    # sample(input_path, output_path_10w_sample, 100000)
    # 5. 生成不重复的数据
    # input_path = "./table/trip_data_1_filter_10w.npy"
    # output_path = "./table/trip_data_1_10w_distinct.npy"
    # create_distinct_data(input_path, output_path)
    # 6. Geohash排序数据
    # input_path = "./table/uniform_10000w.npy"
    # output_path = "./index/uniform_sorted.npy"
    # input_path = "./table/normal_10000w.npy"
    # output_path = "./index/normal_sorted.npy"
    # data_precision = 8
    # region = Region(0, 1, 0, 1)
    # input_path = "./table/trip_data_1_filter.npy"
    # output_path = "./index/nyct_sorted.npy"
    # input_path = "./table/trip_data_1_filter_10w.npy"
    # output_path = "./index/nyct_10w_sorted.npy"
    # data_precision = 6
    # region = Region(40, 42, -75, -73)
    # geohash_and_sort(input_path, output_path, data_precision, region)
    # 7. 生成索引列
    # output_path = "./table/trip_data_1_filter.npy"
    # output_path = "./table/trip_data_1_filter_10w.npy"
    # output_path = "./table/normal_10000w.npy"
    # output_path = "./table/normal_10w.npy"
    # output_path = "./table/uniform_10000w.npy"
    # output_path = "./table/uniform_10w.npy"
    # first_key = 0
    # output_path = "./table/trip_data_2_filter.npy"
    # first_key = 14507253
    # output_path = "./table/trip_data_2_filter_10w.npy"
    # first_key = 100000
    # add_key_field(output_path, output_path, first_key)
    # 1. 生成point检索范围
    # input_path = './table/uniform_10000w.npy'
    # output_path = './query/point_query_uniform.npy'
    # input_path = './table/normal_10000w.npy'
    # output_path = './query/point_query_normal.npy'
    # input_path = './table/trip_data_1_filter.npy'
    # output_path = './query/point_query_nyct.npy'
    # query_number_limit = 1000
    # create_point_query(input_path, output_path, query_number_limit)
    # 2. 生成range检索范围
    # range_ratio_list = [0.001, 0.005, 0.01, 0.015, 0.02]
    # output_path = './query/range_query_uniform.npy'
    # output_path = './query/range_query_normal.npy'
    # data_range = [0, 1, 0, 1]
    # output_path = './query/range_query_nyct.npy'
    # data_range = [40, 42, -75, -73]
    # query_number_limit = 1000
    # create_range_query(output_path, data_range, query_number_limit, range_ratio_list)
    # 3.生成knn检索范围
    # input_path = './table/uniform_10000w.npy'
    # output_path = './query/knn_query_uniform.npy'
    # input_path = './table/normal_10000w.npy'
    # output_path = './query/knn_query_normal.npy'
    # input_path = './table/trip_data_1_filter.npy'
    # output_path = './query/knn_query_nyct.npy'
    # query_number_limit = 1000
    # n_list = [4, 8, 16, 32, 64]
    # create_knn_query(input_path, output_path, query_number_limit, n_list)

    # 实验结果日志转csv，把result里的所有非csv文件转成csv
    input_path = "../result"
    output_path = "../result"
    log_to_csv(input_path, output_path)
