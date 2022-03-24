import csv
import json
import logging
import multiprocessing
import os

import pandas
import pandas as pd

from src.spatial_index.common_utils import Point, Region


def count_csv(path):
    with open(path) as f1:
        reader = csv.reader(f1)
        count = 0
        for line in reader:
            count += 1
        return count


def filter_row_from_csv(input_path, output_path, lines_limit, range_limit=None):
    with open(input_path) as f1, open(output_path, 'w', newline='') as f2:
        count = 0
        reader = csv.reader(f1)
        writer = csv.writer(f2)
        for line in reader:
            if lines_limit and count > lines_limit:
                break

            if count == 0:
                writer.writerow(["index", "x", "y"])
            if count != 0:
                pickup_lng = line[10]
                pickup_lat = line[11]
                if range_limit and range_limit.contain(Point(float(pickup_lng), float(pickup_lat))) is False:
                    continue
                else:
                    target_row = [count - 1, pickup_lng, pickup_lat]
                    writer.writerow(target_row)
            count += 1


def get_region(input_path):
    df = pandas.read_csv(input_path)
    print("Region: %f, %f, %f, %f" % (df.y.min(), df.y.max(), df.x.min(), df.x.max()))


def sample_from_csv(input_path, output_path, lines_limit, range_limit=None):
    df = pandas.read_csv(input_path)
    df["contain"] = df.apply(
        lambda x: range_limit.contain(Point(float(x["x"]), float(x["y"]))), axis=1)
    df = df[df["contain"]]
    df = df.sample(n=lines_limit, random_state=1)
    df = df.reset_index()
    df = df.drop(columns={"index", "contain"})
    df.to_csv(output_path, index_label="index")


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
    multiprocessing.set_start_method('spawn')  # 解决CUDA_ERROR_NOT_INITIALIZED报错
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
    index_list = pd.read_csv(r'D:\Code\Paper\st-learned-index\src\spatial_index\model\gm_index_10w\point_list.csv',
                             float_precision='round_trip', header=None)
    data_list = pd.read_csv(r'D:\Code\Paper\st-learned-index\data\trip_data_1_100000.csv', float_precision='round_trip')
    gm_result_list = r'D:\Code\Paper\st-learned-index\src\spatial_index\model\gm_index_10w\knn_query_result.csv'
    r_result_list = r'D:\Code\Paper\st-learned-index\src\spatial_index\model\rtree_10w\knn_query_result.csv'
    with open(gm_result_list) as f1, open(r_result_list) as f2:
        count = 0
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
    # 数据来源：从http://www.andresmh.com/nyctaxitrips/下载trip_data.7z，拿到其中的一月份数据csv
    # 数据总记录数：14776616，region内14507253
    # csv文件size：2459600863字节=2.29GB
    # input_path = "../../data/trip_data_1.csv"
    # output_path = "../../data/trip_data_1_filter.csv"
    # 1. 生成数据
    # filter_row_from_csv(input_path, output_path, None, Region(40, 42, -75, -73))
    # 输出数据spatial scope
    # 40.016666, 41.933331, -74.990433, -73.000938
    # input_path = "../../data/trip_data_1_filter.csv"
    # get_region(input_path)
    # 2. 生成100000的数据
    # input_path = "../../data/trip_data_1_filter.csv"
    # output_path_100000_sample = '../../data/trip_data_1_100000.csv'
    # sample_from_csv(input_path, output_path_100000_sample, 100000, Region(40, 42, -75, -73))
    # 3. 生成point检索范围
    # output_path_100000_sample = '../../data/trip_data_1_100000.csv'
    # output_path_point_query_csv = '../../data/trip_data_1_point_query.csv'
    # point_limit = 10000
    # create_point_from_csv(output_path_100000_sample, output_path_point_query_csv, point_limit)
    # 4. 生成range检索范围
    # output_path_100000_sample = '../../data/trip_data_1_100000.csv'
    # output_path_range_query_csv = '../../data/trip_data_1_range_query.csv'
    # output_path_range_query_log = '../../data/trip_data_1_range_query.log'
    # window_limit = 100
    # print_window_from_csv_to_log(output_path_100000_sample, output_path_range_query_log, window_limit, 6)
    # create_window_from_log_to_csv(output_path_range_query_log, output_path_range_query_csv)
    # 5.生成knn检索范围
    # output_path_100000_sample = '../../data/trip_data_1_100000.csv'
    # output_path_knn_query_csv = '../../data/trip_data_1_knn_query.csv'
    # knn_point_limit = 1000
    # knn_n_limit = 10
    # create_knn_from_csv(output_path_100000_sample, output_path_knn_query_csv, knn_point_limit, knn_n_limit)

    # 确定knn找到的数据对不对
    check_knn()
