import math
from collections import deque
from itertools import chain
from reprlib import repr
from sys import getsizeof, stderr

import matplotlib.pyplot as plt
import numpy as np


class Point:
    def __init__(self, lng, lat, geohash=None, key=None):
        self.lng = lng
        self.lat = lat
        self.geohash = geohash
        self.key = key

    def __eq__(self, other):
        if other.lng == self.lng and other.lat == self.lat:
            return True
        else:
            return False

    def __str__(self):
        return "Point({0}, {1}, {2})".format(self.lng, self.lat, self.key)

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
        # 优化: math.sqrt->**0.5
        return ((self.lng - other.lng) ** 2 + (self.lat - other.lat) ** 2) ** 0.5

    def distance_pow(self, other):
        """
        计算两点距离的平方
        :param other:
        :return: distance ** 2
        """
        return (self.lng - other.lng) ** 2 + (self.lat - other.lat) ** 2


def intersect(window, other, cross=False):
    """
    a和b相交：两个矩形中心点的xy距离 <= 两个矩形xy边长之和
    a包含b：两个矩形中心点的xy距离 <= 两个矩形xy边长之差(a-b)
    # b包含a：两个矩形中心点的xy距离 <= 两个矩形xy边长之差(b-a)
    :param cross: 是否返回相交部分的region
    :return: 1=intersect, 2=self contain other, 3=other contain self
    """
    center_distance_x = abs(window[2] + window[3] - other[2] - other[3])
    edge_sum_x = window[3] - window[2] + other[3] - other[2]
    if center_distance_x <= edge_sum_x:
        center_distance_y = abs(window[0] + window[1] - other[0] - other[1])
        edge_sum_y = window[1] - window[0] + other[1] - other[0]
        if center_distance_y <= edge_sum_y:
            edge_divide_x = window[3] - window[2] - other[3] + other[2]
            if center_distance_x <= edge_divide_x:
                edge_divide_y = window[1] - window[0] - other[1] + other[0]
                if center_distance_y <= edge_divide_y:
                    if cross:
                        return 2, None
                    else:
                        return 2
            edge_divide_x2 = other[3] - other[2] - window[3] + window[2]
            if center_distance_x <= edge_divide_x2:
                edge_divide_y2 = other[1] - other[0] - window[1] + window[0]
                if center_distance_y <= edge_divide_y2:
                    if cross:
                        return 3, None
                    else:
                        return 3
            if cross:
                return 1, Region(max(window[0], other[0]), min(window[1], other[1]), max(window[2], other[2]),
                                 min(window[3], other[3]))
            else:
                return 1
    if cross:
        return 0, None
    else:
        return 0


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

    def contain(self, point):
        return self.bottom == point.lat or self.left == point.lng or (
                self.up > point.lat > self.bottom and self.right > point.lng > self.left)

    def contain_and_border_by_point(self, point):
        return self.up >= point.lat >= self.bottom and self.right >= point.lng >= self.left

    def contain_and_border(self, lng, lat):
        return self.up >= lat >= self.bottom and self.right >= lng >= self.left

    def within_distance(self, point, distance):
        if point.lng >= self.right:
            if point.lat >= self.up:
                return point.distance(Point(self.right, self.up)) <= distance
            elif self.bottom < point.lat < self.up:
                return point.lng - self.right <= distance
            else:
                return point.distance(Point(self.right, self.bottom)) <= distance
        elif self.left < point.lng < self.right:
            if point.lat <= self.bottom:
                return self.bottom - point.lat <= distance
            elif self.bottom < point.lat < self.up:
                return True
            else:
                return point.lat - self.up <= distance
        else:
            if point.lat <= self.bottom:
                return point.distance(Point(self.left, self.bottom)) <= distance
            elif self.bottom < point.lat < self.up:
                return self.left - point.lng <= distance
            else:
                return point.distance(Point(self.left, self.up)) < distance

    def within_distance_pow(self, point, distance_pow):
        if point.lng >= self.right:
            if point.lat >= self.up:
                return point.distance_pow(Point(self.right, self.up)) <= distance_pow
            elif self.bottom < point.lat < self.up:
                return (point.lng - self.right) ** 2 <= distance_pow
            else:
                return point.distance_pow(Point(self.right, self.bottom)) <= distance_pow
        elif self.left < point.lng < self.right:
            if point.lat <= self.bottom:
                return (self.bottom - point.lat) ** 2 <= distance_pow
            elif self.bottom < point.lat < self.up:
                return True
            else:
                return (point.lat - self.up) ** 2 <= distance_pow
        else:
            if point.lat <= self.bottom:
                return point.distance_pow(Point(self.left, self.bottom)) <= distance_pow
            elif self.bottom < point.lat < self.up:
                return (self.left - point.lng) ** 2 <= distance_pow
            else:
                return point.distance_pow(Point(self.left, self.up)) < distance_pow

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

    def get_min_distance_pow_by_point_list(self, point: list):
        """
        计算点到region的距离，如果点在region内，则为0
        :param point:
        :return:
        """
        if point[0] >= self.right:
            if point[1] >= self.up:
                return (self.right - point[0]) ** 2 + (self.up - point[1]) ** 2
            elif self.bottom < point[1] < self.up:
                return (point[0] - self.right) ** 2
            else:
                return (self.right - point[0]) ** 2 + (self.bottom - point[1]) ** 2
        elif self.left < point[0] < self.right:
            if point[1] <= self.bottom:
                return (self.bottom - point[1]) ** 2
            elif self.bottom < point[1] < self.up:
                # return max(self.bottom - point[1], point[1] - self.up, self.left - point[0], point[0] - self.right)
                return 0
            else:
                return (point[1] - self.up) ** 2
        else:
            if point[1] <= self.bottom:
                return (self.left - point[0]) ** 2 + (self.bottom - point[1]) ** 2
            elif self.bottom < point[1] < self.up:
                return (self.left - point[0]) ** 2
            else:
                return (self.left - point[0]) ** 2 + (self.up - point[1]) ** 2

    def up_right_less(self, i):
        self.up -= i
        self.right -= i

    def up_right_less_region(self, i):
        self.up -= i
        self.right -= i
        return self

    def up_right_more_region(self, i):
        self.up += i
        self.right += i
        return self

    def split(self):
        """
        一分四
        """
        y_center = (self.up + self.bottom) / 2
        x_center = (self.left + self.right) / 2
        LB = Region(self.bottom, y_center, self.left, x_center)
        RB = Region(self.bottom, y_center, x_center, self.right)
        LU = Region(y_center, self.up, self.left, x_center)
        RU = Region(y_center, self.up, x_center, self.right)
        return LB, RB, LU, RU

    def clip_region(self, region, precision):
        """
        把region剪到自己的范围内，precision是为了右上角会超出编码长度，所以往左下偏移
        """
        if region[0] < self.bottom:
            region[0] = self.bottom
        if region[1] > self.up:
            region[1] = self.up - pow(10, -precision)
        if region[2] < self.left:
            region[2] = self.left
        if region[3] > self.right:
            region[3] = self.right - pow(10, -precision)


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


def binary_search_duplicate(nums, field, x, left, right):
    """
    二分查找 + 对象 + 允许重复
    """
    result = []
    while left <= right:
        mid = (left + right) // 2
        if nums[mid][field] == x:
            result.append(mid)
            mid_left = mid - 1
            while mid_left >= left and nums[mid_left][field] == x:
                result.append(mid_left)
                mid_left -= 1
            mid_right = mid + 1
            while mid_right <= right and nums[mid_right][field] == x:
                result.append(mid_right)
                mid_right += 1
            return result
        elif nums[mid][field] < x:
            left = mid + 1
        else:
            right = mid - 1
    return result


def binary_search_less_max(nums, field, x, left, right):
    """
    二分查找 + 找比x小的最大值
    优化: 循环->二分:15->1
    """
    while left <= right:
        mid = (left + right) // 2
        if nums[mid][field] == x:
            return mid
        elif nums[mid][field] < x:
            left = mid + 1
        else:
            right = mid - 1
    return right


def binary_search_less_max_duplicate(nums, x, left, right):
    """
    二分查找 + 不超过x的最大值 + 允许重复
    """
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == x:
            mid += 1
            while mid <= right and nums[mid] == x:
                mid += 1
            return mid
        elif nums[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return right + 1


def biased_search_duplicate(nums, field, x, mid, left, right):
    """
    二分查找 + 对象 + biased
    如果pre不在[left, right]里，会变慢
    """
    result = []
    while left <= right:
        if nums[mid][field] == x:
            result.append(mid)
            mid_left = mid - 1
            while mid_left >= left and nums[mid_left][field] == x:
                result.append(mid_left)
                mid_left -= 1
            mid_right = mid + 1
            while mid_right <= right and nums[mid_right][field] == x:
                result.append(mid_right)
                mid_right += 1
            return result
        elif nums[mid][field] < x:
            left = mid + 1
        else:
            right = mid - 1
        mid = (left + right) // 2
    return result


def biased_search_less_max_duplicate(nums, field, x, mid, left, right):
    """
    二分查找 + 对象 + biased + 不超过x的最大值 + 允许重复
    """
    while left <= right:
        if nums[mid][field] == x:
            mid += 1
            while mid <= right and nums[mid][field] == x:
                mid += 1
            return mid
        elif nums[mid][field] < x:
            left = mid + 1
        else:
            right = mid - 1
        mid = (left + right) // 2
    return right + 1


def biased_search_almost(nums, field, x, mid, left, right):
    """
    二分查找 + 对象 + biased + 查找不超过x的数量 + 允许重复
    """
    result = []
    left_store = left
    right_store = right
    while left <= right:
        if nums[mid][field] == x:
            result.append(mid)
            mid_left = mid - 1
            while mid_left >= left and nums[mid_left][field] == x:
                result.append(mid_left)
                mid_left -= 1
            mid_right = mid + 1
            while mid_right <= right and nums[mid_right][field] == x:
                result.append(mid_right)
                mid_right += 1
            return result
        elif nums[mid][field] < x:
            left = mid + 1
        else:
            right = mid - 1
        mid = (left + right) // 2
    if right < left_store:
        return [left_store]
    if left > right_store:
        return [right_store]
    return [right] if nums[left][field] - x > x - nums[right][field] else [left]


def interpolation_search_less_max(nums, field, x, left, right):
    """
    插入查找 + 对象
    """
    while left < right:
        mid = left + (right - left) * (x - nums[left]) / (nums[right] - nums[left])
        if nums[mid][field] <= x:
            left = mid
        elif nums[mid][field] > x:
            right = mid - 1
    return right


def partition(nums, field, left, right):
    pivot, j = nums[left][field], left
    for i in range(left + 1, right + 1):
        if nums[i][field] <= pivot:
            j += 1
            nums[j], nums[i] = nums[i], nums[j]
    nums[left], nums[j] = nums[j], nums[left]
    return j


def quick_sort(nums, field, left, right):
    """
    快速排序
    """
    if left < right:
        m = partition(nums, field, left, right)
        quick_sort(nums, field, left, m - 1)
        quick_sort(nums, field, m + 1, right)


def quick_sort_n(nums, field, n, left, right):
    """
    快速排序使得前n个数为最小数
    """
    if left < right:
        m = partition(nums, field, left, right)
        if m > n:
            quick_sort_n(nums, field, n, left, m - 1)
        else:
            quick_sort_n(nums, field, n, m + 1, n)


def normalize_input(na):
    min_v = na.min(axis=0)
    max_v = na.max(axis=0)
    if max_v == min_v:
        return na - min_v, min_v, max_v
    else:
        return (na - min_v) / (max_v - min_v) - 0.5, min_v, max_v


def normalize_output(na):
    min_v = na.min(axis=0)
    max_v = na.max(axis=0)
    if max_v == min_v:
        return na, min_v, max_v
    else:
        return (na - min_v) / (max_v - min_v), min_v, max_v


def normalize_input_minmax(value, min_v, max_v):
    if max_v == min_v:
        return value - min_v
    else:
        return (value - min_v) / (max_v - min_v) - 0.5


def denormalize_output_minmax(value, min_v, max_v):
    if max_v == min_v:
        return min_v
    if value < 0:
        return min_v
    elif value > 1:
        return max_v
    return value * (max_v - min_v) + min_v


def denormalize_outputs_minmax(values, min_v, max_v):
    if max_v == min_v:
        values.fill(min_v)
        return values
    values[values < 0] = 0
    values[values > 1] = 1
    return values * (max_v - min_v) + min_v


def relu(x):
    return np.maximum(0, x)


def elu(x, alpha=1):
    a = x[x > 0]
    b = alpha * (np.exp(x[x < 0]) - 1)
    result = np.concatenate((b, a), axis=0)
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_mbr_by_points(points):
    x_max = x_min = points[0][0]
    y_max = y_min = points[0][1]
    for point in points:
        if y_min > point[1]:
            y_min = point[1]
        elif y_max < point[1]:
            y_max = point[1]
        if x_min > point[0]:
            x_min = point[0]
        elif x_max < point[0]:
            x_max = point[0]
    return [y_min, y_max, x_min, x_max]


def merge_sorted_list(lst1, lst2):
    left = 0
    max_key1 = len(lst1) - 1
    for num2 in lst2:
        right = max_key1
        while left <= right:
            mid = (left + right) // 2
            if lst1[mid][2] == num2[2]:
                left = mid
                break
            elif lst1[mid][2] < num2[2]:
                left = mid + 1
            else:
                right = mid - 1
        lst1.insert(left, num2)
        max_key1 += 1


def plot_ts(ts):
    ts_len = len(ts)
    col = 5
    row = math.ceil(ts_len / col)
    width = 3
    height = 3
    plt.figure(figsize=(width * col, height * row), dpi=80)
    for i in range(ts_len):
        plt.subplot(row, col, i + 1)
        plt.plot(ts[i])
    plt.show()


if __name__ == '__main__':
    ts = [[1, 1, 1]] * 11
    plot_ts(ts)
