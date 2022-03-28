from src.spatial_index.common_utils import get_min_max, binary_search_less_max, Region
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
    def __init__(self, version=None, size=None, threshold_number=None, threshold_length=None, difflen=None,
                 geohash=None, blkregs=None, blknums=None, blkindexes=None, blkghs=None, blkghlens=None):
        # meta page
        self.version = version
        self.threshold_number = threshold_number  # 新增
        self.threshold_length = threshold_length  # 新增
        self.geohash = geohash  # 新增：max_length = geohash.sum_bits
        self.difflen = difflen  # 优化计算所需：data z geohash.sum_bits - blk geohash.sumbits
        self.size = size  # 优化计算所需：blk size - 1
        # regular page
        self.blkghs = blkghs  # 改动：只存单个值，不再是range，对应values
        self.blkghlens = blkghlens  # 新增：blk geohash的实际length
        self.blkregs = blkregs  # BRIN-Spatial也有，blk的region
        self.blknums = blknums  # blk的数据量
        self.blkindexes = blkindexes  # 对应itemoffsets

    @staticmethod
    def init_by_dict(d: dict):
        return ZBRIN(version=d['version'],
                     size=d['size'],
                     threshold_number=d['threshold_number'],
                     threshold_length=d['threshold_length'],
                     difflen=d['difflen'],
                     geohash=d['geohash'],
                     blkregs=d['blkregs'],
                     blknums=d['blknums'],
                     blkindexes=d['blkindexes'],
                     blkghs=d['blkghs'],
                     blkghlens=d['blkghlens'])

    def save_to_dict(self):
        return {
            'version': self.version,
            'size': self.size,
            'threshold_number': self.threshold_number,
            'threshold_length': self.threshold_length,
            'difflen': self.difflen,
            'geohash': self.geohash,
            'blkregs': self.blkregs,
            'blknums': self.blknums,
            'blkindexes': self.blkindexes,
            'blkghs': self.blkghs,
            'blkghlens': self.blkghlens,
        }

    def build_by_quadtree(self, quad_tree, geohash_length):
        """
        通过四叉树构建block range
        example:
            quad_tree = QuadTree(region=region, threshold_number=threshold_number, data_precision=data_precision)
            quad_tree.build(data, z=True)
            quad_tree.geohash(self.geohash)
            split_data = quad_tree.leaf_nodes
            zbrin = ZBRIN()
            zbrin.build(quad_tree, geohash_length)
        """
        split_data = quad_tree.leaf_nodes
        self.size = len(split_data) - 1
        self.threshold_number = quad_tree.threshold_number
        self.threshold_length = quad_tree.max_depth * 2
        self.blkregs = [item["region"] for item in split_data]
        self.blknums = [len(item["items"]) for item in split_data]
        self.blkindexes = [get_min_max([point.index for point in item["items"]]) for item in split_data]
        self.blkghlens = [len(item["geohash"]) for item in split_data]
        self.geohash = Geohash(sum_bits=max(self.blkghlens), region=quad_tree.region)
        self.difflen = geohash_length - self.geohash.sum_bits
        self.blkghs = [self.geohash.geohash_to_int(item["geohash"]) for item in split_data]

    def build(self, geohash_list, geohash_length, region, threshold_number, data_precision):
        """
        构建SBRIN
        1. 初始化第一个block
        2. 用堆栈存储block，对每个block进行分裂，直到block的number不超过threshold_number
        3. 把不超过的block加入结果list，加入的时候顺序为[左上，右下，左上，右上]的逆序，因为堆栈
        4. 从结果list存储为SBRIN
        """
        self.threshold_number = threshold_number
        self.threshold_length = region.get_max_depth_by_region_and_precision(precision=data_precision) * 2
        N = len(geohash_list)
        # 初始化第一个block
        init_block_list = [(0, 0, region, N, (0, N - 1))]
        result_block_list = []
        # 开始分裂
        while len(init_block_list):
            cur_block = init_block_list.pop(-1)
            # 如果number不超过threshold_number则为结果block
            if cur_block[3] <= threshold_number:
                result_block_list.append(cur_block)
                continue
            else:  # 如果number超过threshold_number则开始分裂
                child_region = cur_block[2].split()
                left_index = cur_block[4][0]
                right_index = cur_block[4][1]
                tmp_left_index = left_index
                child_block_list = [None] * 4
                # 1. length生成
                length = cur_block[1] + 2
                for i in range(4):
                    # 2. value生成
                    value = (cur_block[0] << 2) + i
                    # 3. region生成
                    region = child_region[i]
                    # 4. 数据继承：blknum和itemoffset生成
                    # 计算right bound
                    min_geohash_value = value + 1 << geohash_length - length
                    # 找到right_bound对应的index
                    tmp_right_index = binary_search_less_max(geohash_list, min_geohash_value, tmp_left_index,
                                                             right_index)
                    child_block_list[i] = (value, length, region, tmp_right_index - tmp_left_index + 1,
                                           (tmp_left_index, tmp_right_index))
                    tmp_left_index = tmp_right_index + 1
                init_block_list.extend(child_block_list[::-1])  # 倒着放入init中，保持顺序
        # 根据result_block_list转为SBRIN存储
        self.blkghlens = [blk[1] for blk in result_block_list]
        self.blkregs = [Region.up_right_less_region(blk[2], pow(10, -data_precision - 1)) for blk in result_block_list]
        self.blknums = [blk[3] for blk in result_block_list]
        self.blkindexes = [blk[4] for blk in result_block_list]
        max_geohash_length = max(self.blkghlens)
        self.blkghs = [blk[0] << max_geohash_length - blk[1] for blk in result_block_list]
        self.geohash = Geohash(sum_bits=max_geohash_length, region=region)
        self.difflen = geohash_length - self.geohash.sum_bits
        self.size = len(result_block_list) - 1

    def point_query(self, point):
        """
        根据z找到所在的blk的index
        1. 计算z对应到blk的geohash_int
        2. 找到比geohash_int小的最大值即为z所在的blk index
        """
        return binary_search_less_max(self.blkghs, point >> self.difflen, 0, self.size)

    def range_query_old(self, point1, point2, window):
        """
        根据z1/z2找到之间所有blk的index以及blk和window的相交关系
        1. 使用point_query查找z1和z2所在block的index
        2. 判断window是否包含这些block之前的region相交或包含
        3. 返回index1, index2, [[index, intersect/contain]]
        TODO: intersect函数还可以改进，改为能判断window对于region的上下左右关系
        """
        i = binary_search_less_max(self.blkghs, point1 >> self.difflen, 0, self.size)
        j = binary_search_less_max(self.blkghs, point2 >> self.difflen, i, self.size)
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
        origin_geohash_list = self.geohash.ranges_by_int(point1 >> self.difflen, point2 >> self.difflen)
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
        origin_geohash_list = self.geohash.ranges_by_int(point1 >> self.difflen, point2 >> self.difflen)
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
