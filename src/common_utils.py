import csv
import random
import time

import morton
import pandas as pd


class Point:
    def __init__(self, lng, lat, z=None, index=None):
        self.lng = lng
        self.lat = lat
        self.z = z
        self.index = index

    def __eq__(self, other):
        if other.lng == self.lng and other.lat == self.lat:
            return True
        else:
            return False

    def __str__(self):
        return "Point({0}, {1}, {2})".format(self.lng, self.lat, self.index)

    def near(self, other):
        """
        近似相等，只要整数部分一致即可
        :param other:
        :return:
        """
        if int(other.lng) == int(self.lng) and int(other.lat) == int(self.lat):
            return True
        else:
            return False

    def compute_z(self):
        self.z = ZOrder().point_to_z(self.lng, self.lat)


class Region:
    def __init__(self, bottom, up, left, right):
        self.bottom = bottom
        self.up = up
        self.left = left
        self.right = right

    def contain(self, point):
        return self.up >= point.lat >= self.bottom and self.right >= point.lng >= self.left


class ZOrder:
    def __init__(self):
        self.morton = morton.Morton(dimensions=2, bits=21)

    def point_to_z(self, lng, lat):
        """
        计算point的z order
        1. degree的point坐标映射到整数，以便计算z order
        zoom的大小取决于point的精度，当前数据的范围是Range(40, 42, -75, -73)，经纬度有6位小数
        则zoom差不多是7位
        2. 使用morton-py.pack(int, int): int计算z order
        :param lng:
        :param lat:
        :return:
        """
        zoom = 1000000
        lng_zoom = int((lng - -75.0) * zoom)
        lat_zoom = int((lat - 40.0) * zoom)
        return self.morton.pack(lng_zoom, lat_zoom)


def create_data(path):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        index = 0
        for i in range(100000):
            lng = random.uniform(-180, 180)
            lat = random.uniform(-90, 90)
            index += 1
            writer.writerow([index, lng, lat])


def create_data_z(input_path, output_path, lng_col, lat_col):
    """
    compute and add z order into file
    1. compute z order by lng and lat
    2. add z order into file
    :param input_path:
    :param output_path:
    :param lng_col:
    :param lat_col:
    :return:
    """
    df = pd.read_csv(input_path, header=None)
    z_order = ZOrder()
    z_values = []
    for i in range(df.count()[0]):
        z_values.append(z_order.point_to_z(df[lng_col][i], df[lat_col][i]))
    df["z_value"] = z_values
    df = df.rename(columns={lng_col: "lng", lat_col: "lat"})
    df = df.sort_values(['z_value'], ascending=[True])
    df = df.reset_index()
    df = df.drop(columns=["index"])
    df.to_csv(output_path, index_label="index", header=None)


def read_data_and_search(path, index, lng_col, lat_col, z_col, index_col):
    index_name = index.name
    data = pd.read_csv(path)
    train_set_point = []
    test_set_point = []
    test_ratio = 0.5  # 测试集占总数据集的比例
    if lng_col and lat_col:
        for i in range(int(data.shape[0])):
            train_set_point.append(Point(data.iloc[i, lng_col], data.iloc[i, lat_col], None, data.iloc[i, index_col]))
    elif z_col:
        for i in range(int(data.shape[0])):
            train_set_point.append(Point(None, None, data.iloc[i, z_col], data.iloc[i, index_col]))
    test_set_point = train_set_point[:int(len(train_set_point) * test_ratio)]

    print("*************start %s************" % index_name)
    print("Start Build")
    start_time = time.time()
    index.build(train_set_point)
    end_time = time.time()
    build_time = end_time - start_time
    print("Build %s time " % index_name, build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    for ind in range(len(test_set_point)):
        err += index.predict(test_set_point[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_point)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_point)
    print("mean error = ", mean_error)
    print("*************end %s************" % index_name)
