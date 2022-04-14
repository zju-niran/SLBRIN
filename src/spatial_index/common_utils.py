import math
from collections import deque
from itertools import chain
from reprlib import repr
from sys import getsizeof, stderr

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

    @staticmethod
    def distance_pow_point_list(point1: list, point2: list):
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

    @staticmethod
    def init_by_dict(d: dict):
        return Point(lng=d['lng'], lat=d['lat'], key=d['key'])


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

    def get_min_distance_pow_by_point_list(self, point: list):
        """
        计算点到region的距离，如果点在region内，则为0
        :param point:
        :return:
        """
        if point[0] >= self.right:
            if point[1] >= self.up:
                return Point.distance_pow_point_list([self.right, self.up], point)
            elif self.bottom < point[1] < self.up:
                return (point[0] - self.right) ** 2
            else:
                return Point.distance_pow_point_list([self.right, self.bottom], point)
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
                return Point.distance_pow_point_list([self.left, self.bottom], point)
            elif self.bottom < point[1] < self.up:
                return (self.left - point[0]) ** 2
            else:
                return Point.distance_pow_point_list([self.left, self.up], point)

    def up_right_less(self, i):
        self.up -= i
        self.right -= i

    @staticmethod
    def up_right_less_region(region, i):
        region.up -= i
        region.right -= i
        return region

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


def binary_search_less_max(nums, field, x, left, right):
    """
    二分查找比x小的最大值
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


def binary_search(nums, field, x, left, right):
    """
    binary search x in nums[left, right]
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


def biased_search_almost(nums, field, x, mid, left, right):
    """
    二分查找，找不到则返回最接近的，值不超过[left, right]
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


def biased_search(nums, field, x, mid, left, right):
    """
    binary search x in nums[left, right], but the first mid is pre
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


def merge_sorted_array(nums1, field, left, right, nums2):
    """
    合并有序数组nums2到有序数组nums的[left, right]之后，并重新排序
    """
    size = len(nums2)
    j = 0
    while j < size:
        if nums1[left][field] >= nums2[j][field]:
            nums1.insert(left, nums2[j])
            j += 1
        left += 1
    while j < size:
        nums1.append(nums2[j])
    new_right = right + size + 1
    del nums1[new_right:new_right + size]


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


def get_nearest_none(lt, pre, left, right):
    """
    找到lt中[left, right]范围内离pre最近的None
    通过offset的增大，找到pre右侧offset的位置或pre左侧offset的None
    如果pre到边界，则left_free和right_free为-1，两者都为-1则return None
    """
    offset = 1
    left_free = 1
    right_free = 1
    if pre < left or pre > right:
        print("invalid pre")
    while True:
        if left_free:
            key = pre + offset
            if key > right:
                left_free = -1
            else:
                if lt[key] is None:
                    return key
        if right_free:
            key = pre - offset
            if key < left:
                right_free = -1
            else:
                if lt[key] is None:
                    return key
        if left_free + right_free == -2:
            return None
        offset += 1


def normalize_input(na):
    min_v = na.min(axis=0)
    max_v = na.max(axis=0)
    if max_v == min_v:
        return na, min_v, max_v
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
        return value
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


def denormalize_diff_minmax(na1, na2, min_v, max_v):
    if max_v == min_v:
        return 0.0, 0.0
    else:
        f1 = np.frompyfunc(denormalize_diff_minmax_child, 4, 1)
        result_na = f1(na1, na2, min_v, max_v).astype('float')
        return result_na.min(), result_na.max()


def denormalize_diff_minmax_child(num1, num2, min_v, max_v):
    if num1 < 0:
        num1 = 0
    elif num1 > 1:
        num1 = 1
    return (num1 - num2) * (max_v - min_v)


def relu(x):
    return np.maximum(0, x)


def elu(x, alpha=1):
    a = x[x > 0]
    b = alpha * (np.exp(x[x < 0]) - 1)
    result = np.concatenate((b, a), axis=0)
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    a = [[5, 5, 5], [1, 1, 1], [2, 2, 2], [5, 5, 5], [4, 4, 4], [3, 3, 3], [0, 0, 0], [5, 5, 5]]
    quick_sort(a, 2, 0, 7)
    print(a)
