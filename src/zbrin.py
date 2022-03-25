from src.spatial_index.common_utils import get_min_max, binary_search_less_max
from src.spatial_index.geohash_utils import Geohash

"""
对brin进行改进，来适应z的索引
1. regular_pages不分page：由于非满四叉树分区数量有限，对应block的数量也有限，因此所有block存储在一个regular_page里
2. regular_pages.values里存储的不再是两个值block，而是一个值：非满四叉树和morton编码适配后，morton排序后的前后分区的morton值是连续的
因此block可以用min_z表示，next_block.min_z = cur_block.max_z + 1
3. regular_page作为brin唯一的结构：由于regular_pages不分page，而且block形成的数据量由数据的空间分布决定，而非数据本身决定，
因此revmap没有存在的意义， meta_page自然也不需要了
"""


class ZBRIN:
    def __init__(self, version=None, size=None, blkregs=None, blknums=None, blkindexes=None,
                 blkghs=None, blkghlens=None, geohash=None, diff_length=None):
        self.version = version
        self.size = size  # blk size - 1
        self.blkregs = blkregs
        self.blknums = blknums
        self.blkindexes = blkindexes
        self.blkghs = blkghs
        self.blkghlens = blkghlens
        self.geohash = geohash
        self.diff_length = diff_length  # data z geohash.sum_bits - blk geohash.sumbits

    @staticmethod
    def init_by_dict(d: dict):
        return ZBRIN(version=d['version'],
                     size=d['size'],
                     blkregs=d['blkregs'],
                     blknums=d['blknums'],
                     blkindexes=d['blkindexes'],
                     blkghs=d['blkghs'],
                     blkghlens=d['blkghlens'],
                     geohash=d['geohash'],
                     diff_length=d['diff_length'])

    def save_to_dict(self):
        return {
            'version': self.version,
            'size': self.size,
            'blkregs': self.blkregs,
            'blknums': self.blknums,
            'blkindexes': self.blkindexes,
            'blkghs': self.blkghs,
            'blkghlens': self.blkghlens,
            'geohash': self.geohash,
            'diff_length': self.diff_length
        }

    def build(self, quad_tree, z_length):
        """
        通过四叉树构建block range
        :param quad_tree:
        :return:
        """
        split_data = quad_tree.leaf_nodes
        self.size = len(split_data) - 1
        self.blkregs = [item["region"] for item in split_data]
        self.blknums = [len(item["items"]) for item in split_data]
        self.blkindexes = [get_min_max([point.index for point in item["items"]]) for item in split_data]
        self.blkghlens = [len(item["geohash"]) for item in split_data]
        self.geohash = Geohash(sum_bits=max(self.blkghlens), region=quad_tree.region)
        self.blkghs = [self.geohash.geohash_to_int(item["geohash"]) for item in split_data]
        self.diff_length = z_length - self.geohash.sum_bits

    def point_query(self, point):
        """
        根据z找到所在的blk的index
        1. 计算z对应到blk的geohash_int
        2. 找到比geohash_int小的最大值即为z所在的blk index
        """
        return binary_search_less_max(self.blkghs, point >> self.diff_length, 0, self.size)

    def range_query_old(self, point1, point2, window):
        """
        根据z1/z2找到之间所有blk的index以及blk和window的相交关系
        1. 使用point_query查找z1和z2所在block的index
        2. 判断window是否包含这些block之前的region相交或包含
        3. 返回index1, index2, [[index, intersect/contain]]
        TODO: intersect函数还可以改进，改为能判断window对于region的上下左右关系
        """
        i = binary_search_less_max(self.blkghs, point1 >> self.diff_length, 0, self.size)
        j = binary_search_less_max(self.blkghs, point2 >> self.diff_length, i, self.size)
        if i == j:
            return [((3, None), i, self.blkindexes[i])]
        else:
            return [(window.intersect(self.blkregs[k]), k, self.blkindexes[k]) for k in range(i, j - 1)]

    def range_query(self, point1, point2):
        """
        根据z1/z2找到之间所有blk的index以及和window的位置关系
        1. 通过geohash_int1/geohash_int2找到window对应的所有origin_geohash和对应window的position
        2. 通过前缀匹配过滤origin_geohash来找到target_geohash
        3. 根据target_geohash分组，并且取最大position
        """
        # 1. get origin geohash and position in the range(geohash_int1, geohash_int2)
        origin_geohash_list = self.geohash.ranges_by_int(point1 >> self.diff_length, point2 >> self.diff_length)
        # 2. get target geohash by prefix match
        size1 = len(origin_geohash_list)
        # 优化: 先用二分找到第一个，107mil->102mil，全部用二分需要348mil
        # i, j = 0, 0
        i, j = 1, binary_search_less_max(self.blkghs, origin_geohash_list[0][0], 0, self.size)
        origin_geohash_list[0][0] = j
        while i < size1:
            if self.blkghs[j] > origin_geohash_list[i][0]:
                origin_geohash_list[i][0] = j - 1
                i += 1
            else:
                j += 1
        # 3. group target geohash and max(position)
        return self.geohash.groupby_and_max(origin_geohash_list)
        # 前缀匹配太慢：时间复杂度=O(len(window对应的geohash个数)*(j-i))
        # return [k
        #         for tmp_geohash in geohash_list
        #         for k in range(i - 1, j)
        #         if self.geohash.compare(self.blkghs[k], tmp_geohash)]

    def knn_query(self, point1, point2, point3):
        """
        根据z1/z2找到之间所有blk的index以及和window的位置关系，并基于和point距离排序
        1. 通过geohash_int1/geohash_int2找到window对应的所有origin_geohash和对应window的position
        2. 通过前缀匹配过滤origin_geohash来找到target_geohash
        3. 根据target_geohash分组，并且取最大position
        """
        # 1. get origin geohash and position in the range(geohash_int1, geohash_int2)
        origin_geohash_list = self.geohash.ranges_by_int(point1 >> self.diff_length, point2 >> self.diff_length)
        # 2. get target geohash by prefix match
        size1 = len(origin_geohash_list)
        i, j = 1, binary_search_less_max(self.blkghs, origin_geohash_list[0][0], 0, self.size)
        origin_geohash_list[0][0] = j
        while i < size1:
            if self.blkghs[j] > origin_geohash_list[i][0]:
                origin_geohash_list[i][0] = j - 1
                i += 1
            else:
                j += 1
        # 3. group target geohash and max(position)
        target_geohash_list = self.geohash.groupby_and_max(origin_geohash_list)
        # 4. compute distance from point and sort by distance
        return sorted([[target_geohash,
                        target_geohash_list[target_geohash],
                        self.blkregs[target_geohash].get_min_distance_pow_by_point_list(point3)]
                       for target_geohash in target_geohash_list], key=lambda x: x[2])
