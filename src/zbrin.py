from src.spatial_index.common_utils import get_min_max, binary_search_less_max
from src.spatial_index.geohash_utils import geohash_to_int, ranges_by_int, groupby_and_max

"""
对brin进行改进，来适应z的索引
1. regular_pages不分page：由于非满四叉树分区数量有限，对应block的数量也有限，因此所有block存储在一个regular_page里
2. regular_pages.values里存储的不再是两个值block，而是一个值：非满四叉树和morton编码适配后，morton排序后的前后分区的morton值是连续的
因此block可以用min_z表示，next_block.min_z = cur_block.max_z + 1
3. regular_page作为brin唯一的结构：由于regular_pages不分page，而且block形成的数据量由数据的空间分布决定，而非数据本身决定，
因此revmap没有存在的意义， meta_page自然也不需要了
"""


class ZBRIN:
    def __init__(self, version=None, size=None, blkregs=None, blknums=None, values=None, indexes=None,
                 geohashs=None, lengths=None, max_length=None):
        self.version = version
        self.size = size
        self.blkregs = blkregs
        self.blknums = blknums
        self.values = values
        self.indexes = indexes
        self.geohashs = geohashs
        self.lengths = lengths
        self.max_length = max_length

    @staticmethod
    def init_by_dict(d: dict):
        return ZBRIN(version=d['version'],
                     size=d['size'],
                     blkregs=d['blkregs'],
                     blknums=d['blknums'],
                     values=d['values'],
                     indexes=d['indexes'],
                     geohashs=d['geohashs'],
                     lengths=d['lengths'],
                     max_length=d['max_length'])

    def save_to_dict(self):
        return {
            'version': self.version,
            'size': self.size,
            'blkregs': self.blkregs,
            'blknums': self.blknums,
            'values': self.values,
            'indexes': self.indexes,
            'geohashs': self.geohashs,
            'lengths': self.lengths,
            'max_length': self.max_length
        }

    def build(self, quad_tree):
        """
        通过四叉树构建block range
        :param quad_tree:
        :return:
        """
        split_data = quad_tree.leaf_nodes
        self.size = len(split_data)
        self.blkregs = [item["region"] for item in split_data]
        self.blknums = [len(item["items"]) for item in split_data]
        self.values = [item["first_z"] for item in split_data]
        self.indexes = [get_min_max([point.index for point in item["items"]]) for item in split_data]
        self.lengths = [len(item["geohash"]) for item in split_data]
        self.max_length = max(self.lengths)
        self.geohashs = [geohash_to_int(item["geohash"], self.max_length) for item in split_data]

    def point_query(self, point):
        """
        query index by z point
        :param point: z
        :return: index
        """
        # for i in range(self.size):
        #     if point < self.values[i]:
        #         break
        # 优化: 8mil=>0.6mil
        index = binary_search_less_max(self.values, point, 0, self.size)
        return index, self.indexes[index]

    def range_query_old(self, point1, point2, window):
        """
        range index by z1/z2 point
        1. 使用point_query查找point1和point2所在block的index
        2. 判断window是否包含这些block之前的region相交或包含
        3. 返回index1, index2, [[index, intersect/contain]]
        TODO: intersect函数还可以改进，改为能判断window对于region的上下左右关系
        """
        # for i in range(self.size):
        #     if point1 < self.values[i]:
        #         break
        # for j in range(i - 1, self.size):
        #     if point2 < self.values[j]:
        #         break
        # 优化：15mil->1.2mil
        i = binary_search_less_max(self.values, point1, 0, self.size - 1)
        j = binary_search_less_max(self.values, point2, i, self.size - 1)
        if i == j:
            return [((3, None), i, self.indexes[i])]
        else:
            return [(window.intersect(self.blkregs[k]), k, self.indexes[k]) for k in range(i, j - 1)]

    def range_query(self, point1, point2):
        """
        range index by geohash_int1/geohash_int2 point
        1. 通过geohash_int1/geohash_int2找到window对应的所有origin_geohash和对应window的position
        2. 通过前缀匹配过滤origin_geohash来找到target_geohash
        3. 根据target_geohash分组，并且取最大position
        """
        # 1. get origin geohash and position in the range(geohash_int1, geohash_int2)
        origin_geohash_list = ranges_by_int(point1, point2, self.max_length)
        # 2. get target geohash by prefix match
        size1 = len(origin_geohash_list)
        # 优化: 先用二分找到第一个，107mil->102mil，全部用二分需要348mil
        # i, j = 0, 0
        i, j = 1, binary_search_less_max(self.geohashs, origin_geohash_list[0][0], 0, self.size)
        origin_geohash_list[0][0] = j
        while i < size1:
            if self.geohashs[j] > origin_geohash_list[i][0]:
                origin_geohash_list[i][0] = j - 1
                i += 1
            else:
                j += 1
        # 3. group target geohash and max(position)
        return groupby_and_max(origin_geohash_list)
        # 前缀匹配太慢：时间复杂度=O(len(window对应的geohash个数)*(j-i))
        # return [k
        #         for tmp_geohash in geohash_list
        #         for k in range(i - 1, j)
        #         if compare(self.geohashs[k], tmp_geohash)]
