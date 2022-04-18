import csv
import json
import logging
import multiprocessing
import os

import numpy as np
import pandas
import pandas as pd

from src.spatial_index.common_utils import Point, Region, quick_sort
from src.spatial_index.geohash_utils import Geohash


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
    df = pd.DataFrame(data_list)
    df = df.sample(n=lines_limit, random_state=1)
    df = df.reset_index()
    df = df.drop(columns={"index"})
    np.save(output_path, df.values)


def create_point_from_csv(input_path, output_path, point_limit):
    df = pandas.read_csv(input_path)
    df_sample = df.sample(n=point_limit, random_state=1)
    df_sample = df_sample.reset_index()
    df_sample = df_sample.drop(columns={"level_0", "index"})
    replicates = []
    for index1, point1 in df_sample.iterrows():
        replicate = 0
        for index, point in df.iterrows():
            if point.x == point1.x and point.y == point1.y:
                replicate += 1
        replicates.append(replicate)
    df_sample["count"] = pd.Series(replicates)
    df_sample.sort_values(by=["count"], ascending=True, inplace=True)
    df_sample.reset_index(drop=True, inplace=True)
    df_sample.to_csv(output_path)


def print_window_from_csv_to_log(input_path, output_path, window_limit, thread_pool_size):
    df = pandas.read_csv(input_path)
    multiprocessing.set_start_method('spawn', force=True)
    pool = multiprocessing.Pool(processes=thread_pool_size)
    df_sample = df.sample(n=window_limit, random_state=1)
    df_sample = df_sample.reset_index()
    df_sample = df_sample.drop(columns={"level_0", "index"})
    for index1, point1 in df_sample.iterrows():
        for index2, point2 in df_sample.iterrows():
            pool.apply_async(print_window_from_csv_child, (output_path, df, index1, point1, index2, point2))
    pool.close()
    pool.join()


def print_window_from_csv_child(output_path, df, index1, point1, index2, point2):
    region = Region.create_region_from_points(point1.x, point1.y, point2.x, point2.y)
    count = 0
    for index, point in df.iterrows():
        if region.contain_and_border(point.x, point.y):
            count += 1
    logging.basicConfig(filename=output_path,
                        level=logging.INFO,
                        format="%(message)s")
    logging.info({"index1": index1,
                  "index2": index2,
                  "bottom": region.bottom,
                  "up": region.up,
                  "left": region.left,
                  "right": region.right,
                  "count": count})


def create_window_from_log_to_csv(log_file, output_path):
    file = open(log_file, "r", encoding='UTF-8')
    results = []
    for line in file:
        result = json.loads(line.replace("\'", '"'))
        results.append(result)
    df = pandas.DataFrame(results)
    df.sort_values(by=["count"], ascending=True, inplace=True)
    df.drop(['index1', 'index2'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path)


def create_knn_from_csv(input_path, output_path, knn_point_limit, knn_n_limit):
    df = pandas.read_csv(input_path)
    df_sample = df.sample(n=knn_point_limit, random_state=1)
    df_sample.drop(columns={"index"}, inplace=True)
    df_sample.reset_index(drop=True, inplace=True)
    result = None
    for i in range(1, knn_n_limit + 1):
        df_sample["n"] = pd.Series([i for j in range(100000)])
        result = result.append(df_sample) if result is not None else df_sample.copy()
    result.reset_index(drop=True, inplace=True)
    result.to_csv(output_path)


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
    data_len = len(data)
    data = [(data[i][0], data[i][1], geohash.encode(data[i][0], data[i][1]), i) for i in range(data_len)]
    import sys
    sys.setrecursionlimit(5000)
    quick_sort(data, 2, 0, data_len - 1)
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
    # input_path = "./table/trip_data_1_filter.npy"
    # output_path_10w_sample = './table/trip_data_1_filter_10w.npy'
    # sample(input_path, output_path_10w_sample, 100000)
    # 5. 生成不重复的数据
    # input_path = "./trip_data_1_10w.npy"
    # output_path = "./trip_data_1_10w_distinct.npy"
    # create_distinct_data(input_path, output_path)
    # 6. Geohash排序数据
    input_path = "./table/trip_data_1_filter.npy"
    output_path = "./index/trip_data_1_filter_sorted.npy"
    geohash_and_sort(input_path, output_path, 6, Region(40, 42, -75, -73))

    # 1. 生成point检索范围
    # output_path_10w_sample = './trip_data_1_10w.csv'
    # output_path_point_query_csv = './trip_data_1_point_query.csv'
    # point_limit = 10000
    # create_point_from_csv(output_path_10w_sample, output_path_point_query_csv, point_limit)
    # 2. 生成range检索范围
    # output_path_10w_sample = './trip_data_1_10w.csv'
    # output_path_range_query_csv = './trip_data_1_range_query.csv'
    # output_path_range_query_log = './trip_data_1_range_query.log'
    # window_limit = 100
    # print_window_from_csv_to_log(output_path_10w_sample, output_path_range_query_log, window_limit, 6)
    # create_window_from_log_to_csv(output_path_range_query_log, output_path_range_query_csv)
    # 3.生成knn检索范围
    # output_path_10w_sample = './query/trip_data_1_10w.csv'
    # output_path_knn_query_csv = './trip_data_1_knn_query.csv'
    # knn_point_limit = 1000
    # knn_n_limit = 10
    # create_knn_from_csv(output_path_10w_sample, output_path_knn_query_csv, knn_point_limit, knn_n_limit)

    # 确定knn找到的数据对不对
    # check_knn()
