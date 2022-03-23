import math
from collections import deque
from itertools import chain
from reprlib import repr
from sys import getsizeof, stderr

import numpy as np


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

    def distance(self, other):
        """
        计算两点距离
        :param other:
        :return: distance
        """
        return math.sqrt((self.lng - other.lng) ** 2 + (self.lat - other.lat) ** 2)

    def distance_pow(self, other):
        """
        计算两点距离的平方
        :param other:
        :return: distance ** 2
        """
        return (self.lng - other.lng) ** 2 + (self.lat - other.lat) ** 2


class Region:
    def __init__(self, bottom, up, left, right):
        self.bottom = bottom
        self.up = up
        self.left = left
        self.right = right

    def __eq__(self, other):
        if other.bottom == self.bottom and other.up == self.up and other.left == self.left and other.right == self.right:
            return True
        else:
            return False

    def intersect(self, other):
        """
        a和b相交：两个矩形中心点的xy距离 <= 两个矩形xy边长之和
        a包含b：两个矩形中心点的xy距离 <= 两个矩形xy边长之差(a-b)
        # b包含a：两个矩形中心点的xy距离 <= 两个矩形xy边长之差(b-a)
        :param other:
        :return: 1=intersect, 2=self contain other, 3=other contain self
        """
        center_distance_x = abs(self.left + self.right - other.left - other.right)
        edge_sum_x = self.right - self.left + other.right - other.left
        if center_distance_x <= edge_sum_x:
            center_distance_y = abs(self.bottom + self.up - other.bottom - other.up)
            edge_sum_y = self.up - self.bottom + other.up - other.bottom
            if center_distance_y <= edge_sum_y:
                edge_divide_x = self.right - self.left - other.right + other.left
                if center_distance_x <= edge_divide_x:
                    edge_divide_y = self.up - self.bottom - other.up + other.bottom
                    if center_distance_y <= edge_divide_y:
                        return 2, None
                # 这里进不来，调用这个方法的地方用if i = j判断过了
                # edge_divide_x2 = other.right - other.left - self.right + self.left
                # if center_distance_x <= edge_divide_x2:
                #     edge_divide_y2 = other.up - other.bottom - self.up + self.bottom
                #     if center_distance_y <= edge_divide_y2:
                #         return 3, None
                return 1, Region(max(self.bottom, other.bottom), min(self.up, other.up), max(self.left, other.left),
                                 min(self.right, other.right))
        return 0, None

    def contain(self, point):
        return self.bottom == point.lat or self.left == point.lng or (
                self.up > point.lat > self.bottom and self.right > point.lng > self.left)

    def contain_and_border_by_point(self, point):
        return self.up >= point.lat >= self.bottom and self.right >= point.lng >= self.left

    def contain_and_border_by_list(self, lt):
        return self.up >= lt[1] >= self.bottom and self.right >= lt[0] >= self.left

    def contain_and_border(self, lng, lat):
        return self.up >= lat >= self.bottom and self.right >= lng >= self.left

    def within_distance(self, point, distance):
        if point.lng >= self.right:
            if point.lat >= self.up:
                return point.distance(Point(self.right, self.up)) <= distance
            elif self.bottom < point.lat < self.up:
                return Point(point.lng, 0).distance(Point(self.right, 0)) <= distance
            else:
                return point.distance(Point(self.right, self.bottom)) <= distance
        elif self.left < point.lng < self.right:
            if point.lat <= self.bottom:
                return Point(0, point.lat).distance(Point(0, self.bottom)) <= distance
            elif self.bottom < point.lat < self.up:
                return True
            else:
                return Point(0, point.lat).distance(Point(0, self.up)) <= distance
        else:
            if point.lat <= self.bottom:
                return point.distance(Point(self.left, self.bottom)) <= distance
            elif self.bottom < point.lat < self.up:
                return Point(point.lng, 0).distance(Point(self.left, 0)) <= distance
            else:
                return point.distance(Point(self.left, self.up)) < distance

    def within_distance_pow(self, point, distance_pow):
        if point.lng >= self.right:
            if point.lat >= self.up:
                return point.distance_pow(Point(self.right, self.up)) <= distance_pow
            elif self.bottom < point.lat < self.up:
                return Point(point.lng, 0).distance_pow(Point(self.right, 0)) <= distance_pow
            else:
                return point.distance_pow(Point(self.right, self.bottom)) <= distance_pow
        elif self.left < point.lng < self.right:
            if point.lat <= self.bottom:
                return Point(0, point.lat).distance_pow(Point(0, self.bottom)) <= distance_pow
            elif self.bottom < point.lat < self.up:
                return True
            else:
                return Point(0, point.lat).distance_pow(Point(0, self.up)) <= distance_pow
        else:
            if point.lat <= self.bottom:
                return point.distance_pow(Point(self.left, self.bottom)) <= distance_pow
            elif self.bottom < point.lat < self.up:
                return Point(point.lng, 0).distance_pow(Point(self.left, 0)) <= distance_pow
            else:
                return point.distance_pow(Point(self.left, self.up)) < distance_pow

    @staticmethod
    def create_region_from_points(x1, y1, x2, y2):
        (bottom, up) = (y2, y1) if y1 > y2 else (y1, y2)
        (left, right) = (x2, x1) if x1 > x2 else (x1, x2)
        return Region(bottom, up, left, right)

    @staticmethod
    def init_by_dict(d: dict):
        return Region(bottom=d['bottom'],
                      up=d['up'],
                      left=d['left'],
                      right=d['right'])

    def get_bits_by_region_and_precision(self, precision):
        """
        从range和数据精度计算morton的bit，最终效果是不重复数据的z不重复
        原理是：不重复数据个数 < region的最短边/10^-precision < 能表示的不重复z个数pow(2, bit)
        => bit = ceil(log2(limit/10^-precision))
        +1是因为后续希望region的角点也能计算z，因此精度+1，来保证region必能把point区分开
        :param precision:
        :return:
        """
        limit = min(self.up - self.bottom, self.right - self.left)
        return math.ceil(math.log(limit / math.pow(10, -precision - 1), 2))

    def get_max_depth_by_region_and_precision(self, precision):
        """
        区别在于精度不加1，来保证最小节点的region宽度>0.000001且再分裂一次就开始小于了
        从range和数据精度计算morton的bit，最终效果是不重复数据的z不重复
        原理是：不重复数据个数 < region的最短边/10^-precision < 能表示的不重复z个数pow(2, bit)
        => bit = ceil(log2(limit/10^-precision))
        :param precision:
        :return:
        """
        limit = min(self.up - self.bottom, self.right - self.left)
        return math.ceil(math.log(limit / math.pow(10, -precision), 2))

    def up_right_less(self, i):
        self.up -= i
        self.right -= i

class ZOrder:
    def __init__(self, data_precision, region):
        self.name = "Z Order"
        self.dimensions = 2
        self.bits = region.get_bits_by_region_and_precision(data_precision)
        self.data_precision = data_precision
        self.region = region
        self.region_width = region.right - region.left
        self.region_height = region.up - region.bottom

        def flp2(x):
            '''Greatest power of 2 less than or equal to x, branch-free.'''
            x |= x >> 1
            x |= x >> 2
            x |= x >> 4
            x |= x >> 8
            x |= x >> 16
            x |= x >> 32
            x -= x >> 1
            return x

        shift = flp2(self.dimensions * (self.bits - 1))
        masks = []
        lshifts = []
        max_value = (1 << (shift * self.bits)) - 1
        while shift > 0:
            mask = 0
            shifted = 0
            for bit in range(self.bits):
                distance = (self.dimensions * bit) - bit
                shifted |= shift & distance
                mask |= 1 << bit << (((shift - 1) ^ max_value) & distance)

            if shifted != 0:
                masks.append(mask)
                lshifts.append(shift)

            shift >>= 1
        self.lshifts = [0] + lshifts
        self.rshifts = lshifts + [0]
        self.max_num = 1 << self.bits
        self.masks = [self.max_num - 1] + masks

    def save_to_dict(self):
        return {
            'name': self.name,
            'data_precision': self.data_precision,
            'region': self.region
        }

    @staticmethod
    def init_by_dict(d: dict):
        return ZOrder(data_precision=d['data_precision'],
                      region=d['region'])

    def split(self, value):
        for o in range(len(self.masks)):
            value = (value | (value << self.lshifts[o])) & self.masks[o]
        return value

    def pack(self, *args):
        code = 0
        for i in range(self.dimensions):
            code |= self.split(args[i]) << i
        return code

    def compact(self, code):
        for o in range(len(self.masks) - 1, -1, -1):
            code = (code | (code >> self.rshifts[o])) & self.masks[o]
        return code

    def unpack(self, code):
        values = []
        for i in range(self.dimensions):
            values.append(self.compact(code >> i))
        return values

    def point_to_z(self, lng, lat):
        """
        计算point的z order
        1. 经纬度都先根据region归一化到0-1，然后缩放到0-2^self.bits
        2. 使用morton-py.pack(int, int): int计算z order，顺序是左下、右下、左上、右上
        :param lng:
        :param lat:
        :return:
        """
        lng_zoom = round((lng - self.region.left) * self.max_num / self.region_width)
        lat_zoom = round((lat - self.region.bottom) * self.max_num / self.region_height)
        return self.pack(lng_zoom, lat_zoom)

    def z_to_point(self, z):
        """
        计算z order的point
        1. 使用morton-py.unpack(int)
        2. 反归一化
        注意：使用round后，z转化的point不一定=计算z的原始point，因为保留有效位数的point和z是多对一的
        如果要一对一，则point_to_z的入口point和z_to_point的出口point都不要用round
        :param z:
        :return:
        """
        lng_zoom, lat_zoom = self.unpack(z)
        lng = lng_zoom * self.region_width / self.max_num + self.region.left
        lat = lat_zoom * self.region_height / self.max_num + self.region.bottom
        return round(lng, self.data_precision), round(lat, self.data_precision)


# python sys.getsizeof无法对自定义类统计内存，提出以下方法
# 代码来自：https://code.activestate.com/recipes/577504
# l342有漏洞：只统计class.__dict__包含的属性，rtree.__dict__不包含bounds等属性，导致内存统计偏小
def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        if not hasattr(o.__class__, '__slots__'):
            if hasattr(o, '__dict__'):
                # no __slots__ *usually* means a __dict__, but some special builtin classes (such as `type(None)`) have neither
                s += sizeof(o.__dict__)
        else:
            s += sum(sizeof(getattr(o, x)) for x in o.__class__.__slots__ if hasattr(o, x))
        return s

    return sizeof(o)


def is_sorted_list(lst):
    """
    判断list是否有序
    """
    return sorted(lst) == lst or sorted(lst, reverse=True) == lst


def nparray_normalize(na):
    """
    对np.array进行最大最小值归一化
    """
    min_v = na.min(axis=0)
    max_v = na.max(axis=0)
    if max_v == min_v:
        return na, None, None
    else:
        return (na - min_v) / (max_v - min_v), min_v, max_v


def normalize_minmax(value, min_v, max_v):
    """
    进行指定最大最小值归一化
    """
    if min_v is None or max_v is None or max_v == min_v:
        return value
    else:
        return (value - min_v) / (max_v - min_v)


def nparray_normalize_reverse_arr(na, min_v, max_v):
    f1 = np.frompyfunc(nparray_normalize_reverse_num, 3, 1)
    return f1(na, min_v, max_v).astype('float')


def nparray_normalize_reverse_num(num, min_v, max_v):
    if min_v is None or max_v is None or max_v == min_v:
        return num
    if num < 0:
        num = 0
    elif num > 1:
        num = 1
    return num * (max_v - min_v) + min_v


def nparray_diff_normalize_reverse_arr(na1, na2, min_v, max_v):
    f1 = np.frompyfunc(nparray_diff_normalize_reverse_num, 4, 1)
    return f1(na1, na2, min_v, max_v).astype('float')


def nparray_diff_normalize_reverse_num(num1, num2, min_v, max_v):
    if min_v is None or max_v is None or max_v == min_v:
        return num1 - num2
    if num1 < 0:
        num1 = 0
    elif num1 > 1:
        num1 = 1
    return (num1 - num2) * (max_v - min_v)


def binary_search_less_max(nums, x, left, right):
    """
    二分查找比x小的最大值
    """
    # TODO：优化，不要这么多判断，最后取left或者right即可
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == x:
            return mid
        elif nums[mid] < x:
            if nums[mid + 1] > x:
                return mid
            left = mid + 1
        else:
            if nums[mid - 1] <= x:
                return mid - 1
            right = mid - 1
    return None


def binary_search(nums, x, left, right):
    """
    binary search x in nums[left, right]
    """
    result = []
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == x:
            result.append(mid)
            mid_left = mid - 1
            while mid_left >= left and nums[mid_left] == x:
                result.append(mid_left)
                mid_left -= 1
            mid_right = mid + 1
            while mid_right <= right and nums[mid_right] == x:
                result.append(mid_right)
                mid_right += 1
            return result
        elif nums[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return result


def biased_search_almost(nums, x, pre, left, right):
    """
    二分查找，找不到则返回最接近的
    """
    mid = pre
    result = []
    while left <= right:
        if nums[mid] == x:
            result.append(mid)
            mid_left = mid - 1
            while mid_left >= left and nums[mid_left] == x:
                result.append(mid_left)
                mid_left -= 1
            mid_right = mid + 1
            while mid_right <= right and nums[mid_right] == x:
                result.append(mid_right)
                mid_right += 1
            return result
        elif nums[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
        mid = (left + right) // 2
    return [right] if nums[left] - x > x - nums[right] else [left]


def biased_search(nums, x, pre, left, right):
    """
    binary search x in nums[left, right], but the first mid is pre
    如果pre不在[left, right]里，会变慢
    """
    mid = pre
    result = []
    while left <= right:
        if nums[mid] == x:
            result.append(mid)
            mid_left = mid - 1
            while mid_left >= left and nums[mid_left] == x:
                result.append(mid_left)
                mid_left -= 1
            mid_right = mid + 1
            while mid_right <= right and nums[mid_right] == x:
                result.append(mid_right)
                mid_right += 1
            return result
        elif nums[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
        mid = (left + right) // 2
    return result


def group_duplicate_list(lt):
    """
    gourp by key1 and max(key2)
    :param lt: [key1, key2]
    :return: [value: [index1, index2]]
    """


def get_min_max(lt):
    if len(lt) == 0:
        return None, None
    min_v = float("inf")
    max_v = float("-inf")
    for i in lt:
        if i > max_v:
            max_v = i
        if i < min_v:
            min_v = i
    return min_v, max_v


if __name__ == '__main__':
    # z = ZOrder(6, Region(40, 42, -75, -73))
    # z1 = z.point_to_z(-74.00001, 40.000001)
    print(binary_search_no_duplicate([0, 2, 4, 6, 8, 10, 11, 13], 7, 0, 7))
