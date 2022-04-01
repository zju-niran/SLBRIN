from src.spatial_index.common_utils import Region, binary_search_less_max
from src.spatial_index.geohash_utils import Geohash


class SBRIN:
    def __init__(self, meta_page=None, regular_pages=None):
        # meta page
        # version: 序号偏移
        # pages_per_range: pages偏移 = (itemoffset - 1) * pagesperrange
        # last_revmap_page: 改动：max_length长度的整型geohash
        # threshold_number: 新增：blk range 分裂的数量阈值
        # threshold_length: 新增：blk range 分裂的geohash长度阈值
        # geohash: 新增：max_length = geohash.sum_bits
        # difflen: 优化计算所需：data z geohash.sum_bits - blk geohash.sumbits
        # size: 优化计算所需：blk size - 1
        self.meta_page = meta_page
        # revmap pages
        # 忽略revmap，pages找blk range的过程，通过itemoffset和pagesperrange直接完成
        # self.revmap_pages = revmap_pages
        # regular pages
        # itemoffet: 序号偏移
        # blknum: pages偏移 = (itemoffset - 1) * pagesperrange
        # value: 改动：max_length长度的整型geohash
        # length: 新增：blk range geohash的实际length
        # number: 新增：blk range的数据量
        # model: 新增：learned indices
        # scope: 优化计算所需：BRIN-Spatial有，blk range的scope
        # index: 优化计算所需：blk range的索引index范围=[blknum * block_size, blknum * block_size + number]
        self.regular_pages = regular_pages

    @staticmethod
    def init_by_dict(d: dict):
        return SBRIN(meta_page=d['meta_page'],
                     regular_pages=d['regular_pages'])

    def build(self, data_list, geohash_length, threshold_number, threshold_length,
              region, data_precision, block_size):
        """
        构建SBRIN
        1. 初始化第一个blk range
        2. 用堆栈存储blk range，对每个blk range进行分裂
        3. 把不超过的blk range加入结果list，加入的时候顺序为[左上，右下，左上，右上]的逆序，因为堆栈
        4. 从结果list创建SBRIN
        5. 重构数据，每个blk range对应threshold_number的数据空间，移到外面做
        """
        # 1. 初始化第一个blk range
        N = len(data_list)
        tmp_stack = [(0, 0, N, (0, N - 1), region)]
        result_list = []
        # 2. 用堆栈存储blk range，对每个blk range进行分裂
        while len(tmp_stack):
            cur = tmp_stack.pop(-1)
            # 如果number超过threshold_number或且length不超过了threshold_length则分裂
            if cur[2] >= threshold_number and cur[1] < threshold_length:
                child_region = cur[4].split()
                left_index = cur[3][0]
                right_index = cur[3][1]
                tmp_left_index = left_index
                child_list = [None] * 4
                # 1. length生成
                length = cur[1] + 2
                for i in range(4):
                    # 2. value生成
                    value = (cur[0] << 2) + i
                    # 3. 数据继承：number生成
                    # 计算right bound
                    min_value = value + 1 << geohash_length - length
                    # 找到right_bound对应的index
                    tmp_right_index = binary_search_less_max(data_list, 2, min_value, tmp_left_index, right_index)
                    child_list[i] = (value, length, tmp_right_index - tmp_left_index + 1,
                                     (tmp_left_index, tmp_right_index), child_region[i])
                    tmp_left_index = tmp_right_index + 1
                tmp_stack.extend(child_list[::-1])  # 倒着放入init中，保持顺序
            else:
                # 3. 把不超过的blk range加入结果list，加入的时候顺序为[左上，右下，左上，右上]的逆序，因为堆栈
                result_list.append(cur)
        # 4. 从结果list创建SBRIN
        # 100为block_size
        pages_per_range = threshold_number // block_size
        max_geohash_length = max([result[1] for result in result_list])
        geohash = Geohash(sum_bits=max_geohash_length, region=region)
        # last_revmap_page理论上是第一个regular_page磁盘位置-1
        self.meta_page = MetaPage(version=1, pages_per_range=pages_per_range, last_revmap_page=0,
                                  threshold_number=threshold_number, threshold_length=threshold_length,
                                  geohash=geohash,
                                  difflen=geohash_length - geohash.sum_bits,
                                  size=len(result_list) - 1)
        self.regular_pages = [RegularPage(itemoffset=i + 1,
                                          blknum=i * pages_per_range,
                                          value=result_list[i][0] << max_geohash_length - result_list[i][1],
                                          length=result_list[i][1],
                                          number=result_list[i][2],
                                          model=None,
                                          scope=Region.up_right_less_region(result_list[i][4],
                                                                            pow(10, -data_precision - 1)),
                                          index=result_list[i][3])
                              for i in range(len(result_list))]
        # revmap_pages理论上要记录每个regular的磁盘位置
        # 5. 重构数据，每个blk range对应threshold_number的数据空间，移到外面做
        # 移到外面做

    def reconstruct_data(self, data_list):
        result_data_list = []
        for regular_page in self.regular_pages:
            result_data_list.extend(data_list[regular_page.index[0]: regular_page.index[1] + 1])
            diff_length = self.meta_page.threshold_number - regular_page.number
            result_data_list.extend([None] * diff_length)

    def point_query(self, point):
        """
        根据z找到所在的blk range的index
        1. 计算z对应到blk的geohash_int
        2. 找到比geohash_int小的最大值即为z所在的blk range
        """
        return self.binary_search_less_max(point >> self.meta_page.difflen, 0, self.meta_page.size)

    def range_query_old(self, point1, point2, window):
        """
        根据z1/z2找到之间所有blk range以及blk range和window的相交关系
        1. 使用point_query查找z1和z2所在blk range
        2. 返回blk range1和blk range2之间的所有blk range，以及他们和window的的包含关系
        TODO: intersect函数还可以改进，改为能判断window对于region的上下左右关系
        """
        i = self.binary_search_less_max(point1 >> self.meta_page.difflen, 0, self.meta_page.size)
        j = self.binary_search_less_max(point2 >> self.meta_page.difflen, i, self.meta_page.size)
        if i == j:
            return [((3, None), self.regular_pages[i])]
        else:
            return [(window.intersect(self.regular_pages[k].scope), self.regular_pages[k]) for k in range(i, j - 1)]

    def range_query(self, point1, point2):
        """
        根据z1/z2找到之间所有blk range的index以及和window的位置关系
        1. 通过geohash_int1/geohash_int2找到window对应的所有origin_geohash和对应window的position
        2. 通过前缀匹配过滤origin_geohash来找到target_geohash
        3. 根据target_geohash分组，并且取最大position
        """
        # 1. get origin geohash and position in the range(geohash_int1, geohash_int2)
        origin_geohash_list = self.meta_page.geohash.ranges_by_int(point1 >> self.meta_page.difflen,
                                                                   point2 >> self.meta_page.difflen)
        # 2. get target geohash by prefix match
        size1 = len(origin_geohash_list)
        # 优化: 先用二分找到第一个，107mil->102mil，全部用二分需要348mil
        # i, j = 0, 0
        i, j = 1, self.binary_search_less_max(origin_geohash_list[0][0], 0, self.meta_page.size)
        origin_geohash_list[0][0] = j
        while i < size1:
            if self.regular_pages[j].value > origin_geohash_list[i][0]:
                origin_geohash_list[i][0] = j - 1
                i += 1
            else:
                j += 1
        # 3. group target geohash and max(position)
        return self.meta_page.geohash.groupby_and_max(origin_geohash_list)
        # 前缀匹配太慢：时间复杂度=O(len(window对应的geohash个数)*(j-i))

    def knn_query(self, point1, point2, point3):
        """
        根据z1/z2找到之间所有blk range的index以及和window的位置关系，并基于和point距离排序
        1. 通过geohash_int1/geohash_int2找到window对应的所有origin_geohash和对应window的position
        2. 通过前缀匹配过滤origin_geohash来找到target_geohash
        3. 根据target_geohash分组，并且取最大position
        """
        # 1. get origin geohash and position in the range(geohash_int1, geohash_int2)
        origin_geohash_list = self.meta_page.geohash.ranges_by_int(point1 >> self.meta_page.difflen,
                                                                   point2 >> self.meta_page.difflen)
        # 2. get target geohash by prefix match
        size1 = len(origin_geohash_list)
        i, j = 1, self.binary_search_less_max(origin_geohash_list[0][0], 0, self.meta_page.size)
        origin_geohash_list[0][0] = j
        while i < size1:
            if self.regular_pages[j].value > origin_geohash_list[i][0]:
                origin_geohash_list[i][0] = j - 1
                i += 1
            else:
                j += 1
        # 3. group target geohash and max(position)
        target_geohash_list = self.meta_page.geohash.groupby_and_max(origin_geohash_list)
        # 4. compute distance from point and sort by distance
        return sorted([[target_geohash,
                        target_geohash_list[target_geohash],
                        self.regular_pages[target_geohash].scope.get_min_distance_pow_by_point_list(point3)]
                       for target_geohash in target_geohash_list], key=lambda x: x[2])

    def binary_search_less_max(self, x, left, right):
        """
        二分查找比x小的最大值
        优化: 循环->二分:15->1
        """
        while left <= right:
            mid = (left + right) // 2
            if self.regular_pages[mid].value == x:
                return mid
            elif self.regular_pages[mid].value < x:
                left = mid + 1
            else:
                right = mid - 1
        return right


class MetaPage:
    def __init__(self, version, pages_per_range, last_revmap_page,
                 threshold_number, threshold_length, geohash, difflen, size):
        # BRIN
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page
        # SBRIN
        self.threshold_number = threshold_number
        self.threshold_length = threshold_length
        self.geohash = geohash
        # For compute
        self.difflen = difflen
        self.size = size

    @staticmethod
    def init_by_dict(d: dict):
        return MetaPage(version=d['version'],
                        pages_per_range=d['pages_per_range'],
                        last_revmap_page=d['last_revmap_page'],
                        threshold_number=d['threshold_number'],
                        threshold_length=d['threshold_length'],
                        geohash=d['geohash'],
                        difflen=d['difflen'],
                        size=d['size'])


class RegularPage:
    def __init__(self, itemoffset, blknum, value, length, number, model, scope, index):
        # BRIN
        self.itemoffset = itemoffset
        self.blknum = blknum
        self.value = value
        # SBRIN
        self.length = length
        self.number = number
        self.model = model
        # For compute
        self.scope = scope
        self.index = index

    @staticmethod
    def init_by_dict(d: dict):
        return RegularPage(itemoffset=d['itemoffset'],
                           blknum=d['blknum'],
                           value=d['value'],
                           length=d['length'],
                           number=d['number'],
                           model=d['model'],
                           scope=d['scope'],
                           index=d['index'])
