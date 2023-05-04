import time
from math import log10

import pandas as pd

from src.utils.common_utils import Point, Region


class Geohash:
    def __init__(self, sum_bits=0, region=Region(-90, 90, -180, 180), data_precision=0):
        self.name = "Geohash"
        self.dimensions = 2
        self.sum_bits = sum_bits
        self.dim_bits = sum_bits // 2
        self.max_num = 1 << self.dim_bits
        self.geohash_template = ['0'] * sum_bits
        self.data_precision = data_precision
        self.region = region
        self.region_width = region.right - region.left
        self.region_height = region.up - region.bottom

    @staticmethod
    def init_by_precision(data_precision, region):
        sum_bits = region.get_bits_by_region_and_precision(data_precision) * 2
        return Geohash(sum_bits, region, data_precision)

    def encode(self, lng, lat):
        """
        计算point的geohash_int
        1. 经纬度都先根据region归一化到0-1，然后缩放到0-2^self.dim_bits
        2. 使用merge_bits把整数的经纬度合并，并转为int，merge的时候是先lat后int，因此顺序是左下、右下、左上、右上
        优化: zorder.pack->int(merge_bits):6->1
        """
        lng_zoom = round((lng - self.region.left) * self.max_num / self.region_width)
        lat_zoom = round((lat - self.region.bottom) * self.max_num / self.region_height)
        return self.merge_bits(lng_zoom, lat_zoom)

    def decode(self, geohash_int):
        """
        计算geohash_int的point
        1. 使用split_bits分开geohash_int为整数的经纬度
        2. 反归一化经纬度，并且round到指定精度
        注意：使用load_index_from_file后，geohash_int转化的point不一定=计算geohash_int的原始point，因为保留有效位数的point和geohash_int是多对一的
        如果要一对一，则encode的入口point和decode的出口point都不要用round
        """
        lng_zoom, lat_zoom = self.split_bits(geohash_int)
        lng = lng_zoom * self.region_width / self.max_num + self.region.left
        lat = lat_zoom * self.region_height / self.max_num + self.region.bottom
        return round(lng, self.data_precision), round(lat, self.data_precision)

    def merge_bits(self, int1, int2):
        self.geohash_template[1::2] = bin(int1)[2:].rjust(self.dim_bits, '0')
        self.geohash_template[0::2] = bin(int2)[2:].rjust(self.dim_bits, '0')
        return int(''.join(self.geohash_template), 2)

    def batch_merge_bits(self, int_range1, int_range2, diff_dim_bits, range_size):
        """
        优化: merge_bits需要range_size次单维度geohash计算，batch后只需要宽*高次单维度geohash计算
        """
        result = [None] * range_size
        i = 0
        geohash_list1 = [bin(int1 << diff_dim_bits)[2:].rjust(self.dim_bits, '0') for int1 in int_range1]
        geohash_list2 = [bin(int2 << diff_dim_bits)[2:].rjust(self.dim_bits, '0') for int2 in int_range2]
        for geohash2 in geohash_list2:
            self.geohash_template[0::2] = geohash2
            for geohash1 in geohash_list1:
                self.geohash_template[1::2] = geohash1
                result[i] = [int(''.join(self.geohash_template), 2), 0]
                i += 1
        return result

    @staticmethod
    def merge_bits_by_length(result, int1, int2, length):
        result[1::2] = bin(int1)[2:].rjust(length, '0')
        result[0::2] = bin(int2)[2:].rjust(length, '0')
        return ''.join(result)

    def split_bits(self, geohash_int):
        geohash = bin(geohash_int)[2:].rjust(self.sum_bits, '0')
        return int(geohash[1::2], 2), int(geohash[0::2], 2)

    def point_to_geohash(self, lng: float, lat: float) -> int:
        lng_zoom = int((lng - self.region.left) * self.max_num / self.region_width)
        lat_zoom = int((lat - self.region.bottom) * self.max_num / self.region_height)
        return self.merge_bits(lng_zoom, lat_zoom)

    def neighbors(self, geohash: str) -> list:
        lng_int = int(geohash[1::2], 2)
        lat_int = int(geohash[0::2], 2)
        length = len(geohash)
        bits_result = [''] * length
        return [self.merge_bits_by_length(bits_result, i, j, length // 2)
                for j in [lat_int - 1, lat_int, lat_int + 1]
                for i in [lng_int - 1, lng_int, lng_int + 1]]

    @staticmethod
    def geohash_to_int(geohash: str, length_origin: int, length_target: int) -> int:
        return int(geohash, 2) << length_target - length_origin

    @staticmethod
    def int_to_geohash(geohash_int: int, length1, length2: int) -> str:
        return bin(geohash_int >> length2 - length1)[2:]

    def ranges_by_int(self, geohash_int1: int, geohash_int2: int, length: int) -> (list, int, int):
        diff_bits = self.sum_bits - length
        diff_dim_bits = self.dim_bits - length // 2
        geohash1 = bin(geohash_int1 >> diff_bits)[2:].rjust(length, '0')
        geohash2 = bin(geohash_int2 >> diff_bits)[2:].rjust(length, '0')
        lng_int1 = int(geohash1[1::2], 2)
        lat_int1 = int(geohash1[0::2], 2)
        lng_int2 = int(geohash2[1::2], 2) + 1
        lat_int2 = int(geohash2[0::2], 2) + 1
        lng_length = lng_int2 - lng_int1
        lat_length = lat_int2 - lat_int1
        result = self.batch_merge_bits(range(lng_int1, lng_int2), range(lat_int1, lat_int2), diff_dim_bits,
                                       lng_length * lat_length)
        # 优化：只计算边界点的grid_num，77mil=>6.7mil
        for j in range(lat_length):
            result[j * lng_length][1] += 2
            result[j * lng_length + lng_length - 1][1] += 1
        for i in range(lng_length):
            result[i][1] += 8
            result[-i - 1][1] += 4
        return sorted(result)
        # return sorted([[int(merge_bits_result(bits_result, i, j, child_length), 2),
        #                 grid_num(i, j, lat_int2, lat_int1, lng_int2, lng_int1)]
        #                for j in range(lat_int1, lat_int2)
        #                for i in range(lng_int1, lng_int2)])

    @staticmethod
    def groupby_and_max(geohash_list: list) -> dict:
        result = {}
        for i in geohash_list:
            # 优化: 4.55mil->2.93mil
            result[i[0]] = result.get(i[0], 0) | i[1]
            # if i[0] not in result.keys():
            #     result[i[0]] = i[1]
            # else:
            #     result[i[0]] |= i[1]
        return result

    @staticmethod
    def grid_num(i, j, lat_int2, lat_int1, lng_int2, lng_int1):
        """
        lu u ru   0110 0100 0101    6  4  5
        l    r => 0010 0000 0001 => 2  0  1
        lb b rb   1010 1000 1001    10 8  9
        b=1000=8, u=0100=4, l=0010=2, r=0001=1
        """
        grid_number = 0
        if j == lat_int1:
            grid_number += 8
        if j == lat_int2:
            grid_number += 4
        if i == lng_int1:
            grid_number += 2
        if i == lng_int2:
            grid_number += 1
        return grid_number

    @staticmethod
    def compare(geohash1: str, geohash2: str) -> bool:
        return geohash1.startswith(geohash2) if len(geohash1) >= len(geohash2) else geohash2.startswith(geohash1)


class Geohash2:
    """
    source code from pypi: python-geohash
    encode：如果找不到c的geohash，就执行encode_base32的代码
    原理：和四叉树一样，经度和维度直接位运算转二进制序列，然后合并
    """

    def encode_base32(self, lng: float, lat: float, precision: int = 12) -> str:
        xprecision = precision + 1
        lat_length = lng_length = int(xprecision * 5 / 2)
        if xprecision % 2 == 1:
            lng_length += 1
        lat = lat / 180.0
        lng = lng / 360.0
        if lat > 0:
            lat = int((1 << lat_length) * lat) + (1 << (lat_length - 1))
        else:
            lat = (1 << lat_length - 1) - int((1 << lat_length) * (-lat))

        if lng > 0:
            lng = int((1 << lng_length) * lng) + (1 << (lng_length - 1))
        else:
            lng = (1 << lng_length - 1) - int((1 << lng_length) * (-lng))
        precision = int((lat_length + lng_length) / 5)
        if lat_length < lng_length:
            a = lng
            b = lat
        else:
            a = lat
            b = lng

        boost = (0, 1, 4, 5, 16, 17, 20, 21)
        ret = ''
        _base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        for i in range(precision):
            ret += _base32[(boost[a & 7] + (boost[b & 3] << 1)) & 0x1F]
            t = a >> 3
            a = b >> 2
            b = t

        ret = ret[::-1]
        return ret[: precision]

    def encode(self, lng: float, lat: float, precision: int = 60) -> str:
        length = precision // 2
        lat = int((1 << length) * (lat + 90) / 180)
        lng = int((1 << length) * (lng + 180) / 360)
        _base32 = ['00', '01', '10', '11']
        ret = ''
        for i in range(length):
            ret += _base32[(lng & 1) + ((lat & 1) << 1)]
            lng = lng >> 1
            lat = lat >> 1
        return ret[::-1][:precision]

    @staticmethod
    def test_python_geohash():
        import geohash
        lnggitude = -5.6
        latitude = 42.6
        hashcode = geohash.encode(latitude, lnggitude, precision=5)
        latitude, lnggitude = geohash.decode(hashcode, delta=False)  # 解码, 返回中间坐标
        latitude, lnggitude, latitude_delta, lnggitude_delta = geohash.decode(hashcode, delta=True)  # 解码，返回中间坐标和半径
        bbox_dict = geohash.bbox(hashcode)  # 边界经纬度，返回四至坐标
        nergnbors_list = geohash.neighbors(hashcode)  # 8个近邻编码
        b = geohash.expand(hashcode)  # 拓展编码 = 8个近邻编码和自己


class Geohash3:
    """
    source code from https://github.com/aseelye/geohash
    原理：和四叉树一样，经度和维度分别二分获得二进制序列，然后合并
    modified: 输出base32改成输出二进制
    """

    def get_bits(self, degrees: float, precision: int, range_ends: int) -> str:
        result = ''
        tup_range = (-range_ends, range_ends)
        for i in range(precision):
            mid = sum(tup_range) / 2
            if degrees > mid:
                result += '1'
                tup_range = (mid, tup_range[1])
            else:
                result += '0'
                tup_range = (tup_range[0], mid)
        return result

    def get_geobits(self, geohash: str) -> str:
        geobits = ''
        for i in geohash:
            try:
                geobits += str(bin(self.dict32[i])[2:].zfill(5))
            except KeyError:
                return "Invalid geohash character.  Use 0-9, b-h, j, k, m, n, p-z."
        return geobits

    def neighbors(self, geohash: str) -> list:
        lng_bits = geohash[::2]
        lat_bits = geohash[1::2]
        lng_int = int(lng_bits, 2)
        lat_int = int(lat_bits, 2)
        length = len(lng_bits)
        hash_len = len(geohash)
        lng_list = [lng_int - 1, lng_int, lng_int + 1]
        lat_list = [lat_int + 1, lat_int, lat_int - 1]
        geo_list = []
        for i in lng_list:
            for j in lat_list:
                geobits = [''] * hash_len
                geobits[::2] = bin(i)[2:].zfill(length)
                geobits[1::2] = bin(j)[2:].zfill(length)
                geo_list.append(''.join(geobits))
        return geo_list

    def encode(self, lng: float, lat: float, precision: int = 60) -> str:
        """
        Encode lng-lat pair to geohash
        :param lng: Longitude
        :param lat: Latitude
        :param precision: Bits of precision.
        :return: Geohash string
        """
        geobits = [''] * precision
        geobits[::2] = self.get_bits(lng, precision // 2, 180)
        geobits[1::2] = self.get_bits(lat, precision // 2, 90)
        return ''.join(geobits)


class Geohash4:
    """
    source code from https://github.com/vinsci/geohash
    原理：和四叉树一样，经度和纬度一起二分，直接形成最终的二进制序列
    modified: change geohash code into 2 bit encode
    """

    def decode_exactly(self, geohash):
        """
        Decode the geohash to its exact values, including the error
        margins of the result.  Returns four float values: latitude,
        lnggitude, the plus/minus error for latitude (as a positive
        number) and the plus/minus error for lnggitude (as a positive
        number).
        """
        lat_interval, lng_interval = (-90.0, 90.0), (-180.0, 180.0)
        lat_err, lng_err = 90.0, 180.0
        is_even = True
        for c in geohash:
            if is_even:  # adds lnggitude info
                lng_err /= 2
                if c == "1":
                    lng_interval = ((lng_interval[0] + lng_interval[1]) / 2, lng_interval[1])
                else:
                    lng_interval = (lng_interval[0], (lng_interval[0] + lng_interval[1]) / 2)
            else:  # adds latitude info
                lat_err /= 2
                if c == "1":
                    lat_interval = ((lat_interval[0] + lat_interval[1]) / 2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0] + lat_interval[1]) / 2)
            is_even = not is_even
        lat = (lat_interval[0] + lat_interval[1]) / 2
        lng = (lng_interval[0] + lng_interval[1]) / 2
        return lat, lng, lat_err, lng_err

    def decode(self, geohash):
        """
        Decode geohash, returning two strings with latitude and lnggitude
        containing only relevant digits and with trailing zeroes removed.
        """
        lat, lng, lat_err, lng_err = self.decode_exactly(geohash)
        # Format to the number of decimals that are known
        lats = "%.*f" % (max(1, round(-log10(lat_err))) - 1, lat)
        lngs = "%.*f" % (max(1, round(-log10(lng_err))) - 1, lng)
        if '.' in lats: lats = lats.rstrip('0')
        if '.' in lngs: lngs = lngs.rstrip('0')
        return lngs, lats

    def encode(self, lnggitude, latitude, precision):
        """
        Encode a position given in float arguments latitude, lnggitude to
        a geohash which will have the character count precision.
        """
        lat_interval, lng_interval = (-90.0, 90.0), (-180.0, 180.0)
        geohash = []
        even = True
        while len(geohash) < precision:
            if even:  # 本来是经度放偶数位，形成经度维度经度维度，但是下面是从左往右下的，所以先写经度
                mid = (lng_interval[0] + lng_interval[1]) / 2
                if lnggitude > mid:
                    geohash += "1"
                    lng_interval = (mid, lng_interval[1])
                else:
                    geohash += "0"
                    lng_interval = (lng_interval[0], mid)
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


def compare_with_python_geohash():
    """
    测试六种Geohash的性能：
    Python-Geohash C create time  9.574317932128907e-07
    My geohash create time  4.040763378143311e-06
    Python-Geohash encode32 create time  7.69151210784912e-06
    Python-Geohash create time  1.0932393074035645e-05
    Geohash3 create time  2.0608618259429933e-05
    Geohash4 create time  2.2692687511444092e-05
    """
    import os
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    path = '../../data/test_x_y_index.csv'
    data = pd.read_csv(path, header=None)
    train_set_point = []
    for i in range(int(data.shape[0])):
        train_set_point.append(Point(data.iloc[i, 1], data.iloc[i, 2], None, data.iloc[i, 0]))
    # python C geohash
    import geohash as pygeohash
    start_time = time.time()
    for ind in range(len(train_set_point)):
        hashcode = pygeohash.encode(train_set_point[ind].lat, train_set_point[ind].lng, precision=12)
    end_time = time.time()
    print(hashcode)
    search_time = (end_time - start_time) / len(train_set_point)
    print("Python-Geohash C create time ", search_time)
    # my geohash
    start_time = time.time()
    geohash = Geohash(60)
    for ind in range(len(train_set_point)):
        hashcode = geohash.point_to_geohash(train_set_point[ind].lng, train_set_point[ind].lat)
    end_time = time.time()
    print(hashcode)
    search_time = (end_time - start_time) / len(train_set_point)
    print("My geohash create time ", search_time)
    # python geohash encode32
    start_time = time.time()
    for ind in range(len(train_set_point)):
        hashcode = Geohash2().encode_base32(train_set_point[ind].lng, train_set_point[ind].lat, precision=12)
    end_time = time.time()
    print(hashcode)
    search_time = (end_time - start_time) / len(train_set_point)
    print("Python-Geohash encode32 create time ", search_time)
    # python geohash
    start_time = time.time()
    for ind in range(len(train_set_point)):
        hashcode = Geohash2().encode(train_set_point[ind].lng, train_set_point[ind].lat, precision=60)
    end_time = time.time()
    print(hashcode)
    search_time = (end_time - start_time) / len(train_set_point)
    print("Python-Geohash create time ", search_time)
    # Geohash3
    start_time = time.time()
    for ind in range(len(train_set_point)):
        hashcode = Geohash3().encode(train_set_point[ind].lng, train_set_point[ind].lat, precision=60)
    end_time = time.time()
    print(hashcode)
    search_time = (end_time - start_time) / len(train_set_point)
    print("Geohash3 create time ", search_time)
    # Geohash4
    start_time = time.time()
    for ind in range(len(train_set_point)):
        hashcode = Geohash4().encode(train_set_point[ind].lng, train_set_point[ind].lat, precision=60)
    end_time = time.time()
    print(hashcode)
    search_time = (end_time - start_time) / len(train_set_point)
    print("Geohash4 create time ", search_time)
    # for ind in range(len(train_set_point)):
    #     hashcode1 = Geohash().encode1(train_set_point[ind].lng, train_set_point[ind].lat, precision=60)
    #     hashcode2 = Geohash().encode4(train_set_point[ind].lng, train_set_point[ind].lat, precision=60)
    #     if hashcode1 != hashcode2:
    #         print(hashcode1)
    #         print(hashcode2)


if __name__ == '__main__':
    # import geohash as pygeohash

    # print(pygeohash.encode(-88.41707557084398, -165.9706735611812, precision=12))
    # print(Geohash(60).point_to_geohash(-165.9706735611812, -88.41707557084398))
    # print(Geohash2().encode(-165.9706735611812, -88.41707557084398, precision=60))
    # print(Geohash2().encode_base32(-165.9706735611812, -88.41707557084398, precision=12))
    # print(Geohash3().encode(-165.9706735611812, -88.41707557084398, precision=60))
    # print(Geohash4().encode(-165.9706735611812, -88.41707557084398, precision=60))
    # print(Geohash(60).neighbors('0011'))
    # print(Geohash3().neighbors('0011'))
    # print(pygeohash.neighbors('023cp0pv4yxb'))
    compare_with_python_geohash()
