import csv
import random
import time
from math import log10

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


class Geohash:
    """
    source code from https://github.com/vinsci/geohash
    modified: change geohash code into 2 bit encode
    """

    def decode_exactly(self, geohash):
        """
        Decode the geohash to its exact values, including the error
        margins of the result.  Returns four float values: latitude,
        longitude, the plus/minus error for latitude (as a positive
        number) and the plus/minus error for longitude (as a positive
        number).
        """
        lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
        lat_err, lon_err = 90.0, 180.0
        is_even = True
        for c in geohash:
            if is_even:  # adds longitude info
                lon_err /= 2
                if c == "1":
                    lon_interval = ((lon_interval[0] + lon_interval[1]) / 2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0] + lon_interval[1]) / 2)
            else:  # adds latitude info
                lat_err /= 2
                if c == "1":
                    lat_interval = ((lat_interval[0] + lat_interval[1]) / 2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0] + lat_interval[1]) / 2)
            is_even = not is_even
        lat = (lat_interval[0] + lat_interval[1]) / 2
        lon = (lon_interval[0] + lon_interval[1]) / 2
        return lat, lon, lat_err, lon_err

    def decode(self, geohash):
        """
        Decode geohash, returning two strings with latitude and longitude
        containing only relevant digits and with trailing zeroes removed.
        """
        lat, lon, lat_err, lon_err = self.decode_exactly(geohash)
        # Format to the number of decimals that are known
        lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
        lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)
        if '.' in lats: lats = lats.rstrip('0')
        if '.' in lons: lons = lons.rstrip('0')
        return lons, lats

    def encode(self, longitude, latitude, precision=12):
        """
        Encode a position given in float arguments latitude, longitude to
        a geohash which will have the character count precision.
        """
        lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
        geohash = []
        even = True
        while len(geohash) < precision:
            if even:  # 本来是经度放偶数位，形成经度维度经度维度，但是下面是从左往右下的，所以先写经度
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if longitude > mid:
                    geohash += "1"
                    lon_interval = (mid, lon_interval[1])
                else:
                    geohash += "0"
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if latitude > mid:
                    geohash += "1"
                    lat_interval = (mid, lat_interval[1])
                else:
                    geohash += "0"
                    lat_interval = (lat_interval[0], mid)
            even = not even
        return ''.join(geohash)

    @staticmethod
    def compare_with_python_geohash():
        """
        对比python-geohash和geohash的encode性能
        Python-Geohash create time  2.742764949798584e-06
        My geohash create time  1.8420519828796385e-05
        """
        import os
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        path = '../../data/test_x_y_index.csv'
        data = pd.read_csv(path, header=None)
        train_set_point = []
        for i in range(int(data.shape[0])):
            train_set_point.append(Point(data.iloc[i, 1], data.iloc[i, 2], None, data.iloc[i, 0]))
        # python geohash
        import geohash as pygeohash
        _base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        _base32_map = {}
        for i in range(len(_base32)):
            _base32_map[_base32[i]] = i
        del i
        start_time = time.time()
        for ind in range(len(train_set_point)):
            hashcode = pygeohash.encode(train_set_point[ind].lat, train_set_point[ind].lng, precision=5)
        end_time = time.time()
        search_time = (end_time - start_time) / len(train_set_point)
        print("Python-Geohash create time ", search_time)
        # my geohash
        start_time = time.time()
        for ind in range(len(train_set_point)):
            hashcode = Geohash().encode(train_set_point[ind].lng, train_set_point[ind].lat, precision=25)
        end_time = time.time()
        search_time = (end_time - start_time) / len(train_set_point)
        print("My geohash create time ", search_time)

    @staticmethod
    def test_python_geohash():
        import geohash
        longitude = -5.6
        latitude = 42.6
        hashcode = geohash.encode(latitude, longitude, precision=5)
        latitude, longitude = geohash.decode(hashcode, delta=False)  # 解码, 返回中间坐标
        latitude, longitude, latitude_delta, longitude_delta = geohash.decode(hashcode, delta=True)  # 解码，返回中间坐标和半径
        bbox_dict = geohash.bbox(hashcode)  # 边界经纬度，返回四至坐标
        nergnbors_list = geohash.neighbors(hashcode)  # 8个近邻编码
        b = geohash.expand(hashcode)  # 拓展编码 = 8个近邻编码和自己


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
    z_values_normalization = []
    for i in range(df.count()[0]):
        z_value = z_order.point_to_z(df[lng_col][i], df[lat_col][i])
        z_values.append(z_value)
    # z归一化
    min_z_value = min(z_values)
    max_z_value = max(z_values)
    for i in range(df.count()[0]):
        z_value_normalization = (z_values[i] - min_z_value) / (max_z_value - min_z_value)
        z_values_normalization.append(z_value_normalization)
    df["z_value"] = z_values
    df["z_value_normalization"] = z_values_normalization
    df = df.rename(columns={lng_col: "lng", lat_col: "lat"})
    df = df.sort_values(['z_value'], ascending=[True])
    df = df.reset_index()
    df = df.drop(columns=["index"])
    df.to_csv(output_path, index_label="index", header=None)


def read_data_and_search(path, index, lng_col, lat_col, z_col, index_col):
    index_name = index.name
    data = pd.read_csv(path, header=None)
    train_set_point = []
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


if __name__ == '__main__':
    geohash = Geohash()
    print(geohash.encode(-5.6, 42.6, precision=25))
    print(geohash.decode('0110111111110000010000010'))
    geohash.compare_with_python_geohash()
