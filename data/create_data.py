import csv
import os

import numpy as np
import pandas
import pandas as pd

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


def get_region(input_path):
    df = pandas.read_csv(input_path)
    print("Region: %f, %f, %f, %f" % (df.y.min(), df.y.max(), df.x.min(), df.x.max()))


def sample(input_path, output_path, lines_limit):
    data_list = np.load(input_path, allow_pickle=True)
    np.random.seed(1)
    sample_key = np.random.randint(0, len(data_list) - 1, size=lines_limit)
    np.save(output_path, data_list[sample_key])


def create_point_query(input_path, output_path, query_number_limit):
    data_list = np.load(input_path, allow_pickle=True)[:, 10:12]
    np.random.seed(1)
    sample_key = np.random.randint(0, len(data_list) - 1, size=query_number_limit)
    np.save(output_path, data_list[sample_key])


def create_range_query(output_path, data_range, query_number_limit, range_ratio_list):
    result = np.empty(shape=(0, 5))
    data_range_width = data_range[3] - data_range[2]
    data_range_height = data_range[1] - data_range[0]
    for range_ratio in range_ratio_list:
        if (1 / range_ratio) ** 2 < query_number_limit:
            raise MyError("range ratio %s is too large with the query number limit %s" %
                          (range_ratio, query_number_limit))
        child_range_width = data_range_width * range_ratio
        child_range_height = data_range_height * range_ratio
        child_range_number_single_dim = int(1 / range_ratio)
        sample_size = min(child_range_number_single_dim, query_number_limit)
        sample_x_key_list = np.random.randint(0, child_range_number_single_dim - 1, size=sample_size)
        sample_y_key_list = np.random.randint(0, child_range_number_single_dim - 1, size=sample_size)
        child_ranges = [[data_range[0] + child_range_height * sample_y_key,
                         data_range[0] + child_range_height * (sample_y_key + 1),
                         data_range[2] + child_range_width * sample_x_key,
                         data_range[2] + child_range_width * (sample_x_key + 1)]
                        for sample_x_key in sample_x_key_list
                        for sample_y_key in sample_y_key_list]
        range_ratio_list = np.array([[range_ratio]] * query_number_limit)
        sample_key_list = np.random.randint(0, sample_size ** 2 - 1, size=query_number_limit)
        child_ranges = np.array(child_ranges)[sample_key_list]
        child_ranges = np.hstack((child_ranges, range_ratio_list))
        result = np.vstack((result, child_ranges))
    np.save(output_path, result)


def create_knn_query(input_path, output_path, query_number_limit, n_list):
    data_list = np.load(input_path, allow_pickle=True)[:, 10:12]
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


def check_knn():
    index_list = pd.read_csv(r'D:\Code\Paper\st-learned-index\src\spatial_index\model\sbrin_10w\point_list.csv',
                             float_precision='round_trip', header=None)
    data_list = pd.read_csv(r'D:\Code\Paper\st-learned-index\data\trip_data_1_10w.csv', float_precision='round_trip')
    gm_result_list = r'D:\Code\Paper\st-learned-index\src\spatial_index\model\sbrin_10w\knn_query_result.csv'
    r_result_list = r'D:\Code\Paper\st-learned-index\src\spatial_index\model\rtree_10w\knn_query_result.csv'
    with open(gm_result_list) as f1, open(r_result_list) as f2:
        list1 = []
        list2 = []
        for line1 in f1.readlines():
            l = []
            line1 = line1[1:-2]
            indexes = line1.split(", ")
            for index1 in indexes:
                index1 = int(index1)
                row = index_list.iloc[index1]
                l.append([row[0], row[1]])
            list1.append(l)
        for line2 in f2.readlines():
            l = []
            line2 = line2[1:-2]
            indexes = line2.split(", ")
            for index2 in indexes:
                index2 = int(index2)
                row = data_list.iloc[index2]
                l.append([row[1], row[2]])
            list2.append(l)
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            if list1[i][j] not in list2[i]:
                print("gm: %s, r: %s" % (list1[i], list2[i]))


def geohash_and_sort(input_path, output_path, data_precision, region):
    geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
    data = np.load(input_path, allow_pickle=True)[:, 10:12]
    data = [(data[i][0], data[i][1], geohash.encode(data[i][0], data[i][1]), i) for i in range(len(data))]
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
    # 数据总记录数：14776615/13990176，region内14507252/13729724
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
    # output_path_uniform = "./table/uniform_10000w.npy"
    # output_path_normal = "./table/normal_10000w.npy"
    # create_data(output_path_uniform, 100000000, [0, 1, 0, 1], 10, 'uniform')
    # create_data(output_path_normal, 100000000, [0, 1, 0, 1], 10, 'normal')
    # plot_npy(output_path_uniform)
    # plot_npy(output_path_normal)
    # 4. 生成10w的数据
    # input_path = "./table/normal_10000w.npy"
    # output_path_10w_sample = './table/normal_10w.npy'
    # sample(input_path, output_path_10w_sample, 100000)
    # 5. 生成不重复的数据
    # input_path = "./trip_data_1_10w.npy"
    # output_path = "./trip_data_1_10w_distinct.npy"
    # create_distinct_data(input_path, output_path)
    # 6. Geohash排序数据
    # input_path = "./table/trip_data_1_filter.npy"
    # output_path = "./index/trip_data_1_filter_sorted.npy"
    # geohash_and_sort(input_path, output_path, 6, Region(40, 42, -75, -73))

    # 1. 生成point检索范围
    input_path = './table/trip_data_1_filter.npy'
    output_path = './query/point_query.npy'
    # input_path = './table/trip_data_1_filter_10w.npy'
    # output_path = './query/point_query_10w.npy'
    query_number_limit = 1000
    selectivity_list = [0.1, 0.5, 1, 1.5, 2]
    create_point_query(input_path, output_path, query_number_limit)
    # 2. 生成range检索范围
    output_path = './query/range_query.npy'
    range_ratio_list = [0.000006, 0.000025, 0.0001, 0.0004, 0.0016]
    # output_path = './query/range_query_10w.npy'
    # range_ratio_list = [0.0025, 0.005, 0.01, 0.02, 0.04]
    data_range = [40, 42, -75, -73]
    query_number_limit = 1000
    create_range_query(output_path, data_range, query_number_limit, range_ratio_list)
    # 3.生成knn检索范围
    input_path = './table/trip_data_1_filter.npy'
    output_path = './query/knn_query.npy'
    # input_path = './table/trip_data_1_filter_10w.npy'
    # output_path = './query/knn_query_10w.npy'
    query_number_limit = 1000
    n_list = [4, 8, 16, 32, 64]
    create_knn_query(input_path, output_path, query_number_limit, n_list)

    # 确定knn找到的数据对不对
    # check_knn()
