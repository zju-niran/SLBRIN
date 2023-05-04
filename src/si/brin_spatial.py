import logging
import math
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.spatial_index import SpatialIndex
from src.utils.common_utils import get_mbr_by_points, intersect, Region
from src.utils.geohash_utils import Geohash

PAGE_SIZE = 4096
RANGE_SIZE = 8 * 4 + 4  # 36
REVMAP_SIZE = 4 + 2  # 6
ITEM_SIZE = 8 * 2 + 4  # 20


class BRINSpatial(SpatialIndex):
    """
    空间块范围索引（BRIN-Spatial）
    Implement from Indexing the pickup and drop-off locations of NYC taxi trips in PostgreSQL - lessons from the road
    """

    def __init__(self, model_path=None, meta=None, block_ranges=None):
        super(BRINSpatial, self).__init__("BRIN Spatial")
        self.index_entries = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # meta page由meta组成
        # version
        # pages_per_range
        # last_revmap_page
        # datas_per_range: 优化计算所需：每个blk的数据容量
        # datas_per_page: 优化计算所需：每个page的数据容量
        # is_sorted: 增加: 优化查询，对blk内的数据按照geohash排序
        # geohash: 增加，is_sorted=True时使用
        self.meta = meta
        # revmap pages由多个revmap分页组成
        # 忽略revmap，pages找block的过程，通过blk的id和pagesperrange直接完成
        # self.revmaps = revmaps
        # regular pages由多个block分页组成
        # blknum: pages偏移 = id * pagesperrange，为便于检索，直接存id
        # value: 改动：blk的MBR
        self.block_ranges = block_ranges
        # for compute
        self.io_cost = 0

    def insert_single(self, point):
        if self.meta.is_sorted:
            # 1. compute geohash from x/y of point
            gh = self.meta.geohash.encode(point[0], point[1])
            # 2. encode p to geohash and create index entry(x, y, geohash, t, pointer)
            point = (point[0], point[1], gh, point[2], point[3])
            # 3. append in xy index
            self.index_entries.append(tuple(point))
            # 3. create tmp blk and sort last blk if point is on the breakpoint
            if point[-1] % self.meta.datas_per_range == 0:
                target_points = sorted(self.index_entries[-self.meta.datas_per_range:], key=lambda x: x[2])
                self.block_ranges[-1].value = get_mbr_by_points(target_points)
                self.index_entries[-self.meta.datas_per_range:] = target_points
                self.create_tmp_blk()
        else:
            # 1. append in xy index
            self.index_entries.append(tuple(point))
            # 2. create tmp blk if point is on the breakpoint
            if point[-1] % self.meta.datas_per_range == 0:
                self.block_ranges[-1].value = get_mbr_by_points(self.index_entries[-self.meta.datas_per_range:])
                self.create_tmp_blk()

    def insert(self, points):
        points = points.tolist()
        for point in points:
            self.insert_single(point)
        # 如果整体插入已经结束，则主动更新tmp br的value
        self.sum_up_tmp_blk()

    def create_tmp_blk(self):
        self.block_ranges.append(BlockRange(blknum=len(self.block_ranges) * self.meta.datas_per_range, value=None))

    def sum_up_tmp_blk(self):
        tmp_blk = self.block_ranges[-1]
        if not tmp_blk.value:
            if self.meta.is_sorted:
                target_points = sorted(self.index_entries[tmp_blk.blknum:], key=lambda x: x[2])
                self.block_ranges[-1].value = get_mbr_by_points(target_points)
                self.index_entries[tmp_blk.blknum:] = target_points
            else:
                tmp_blk.value = get_mbr_by_points(self.index_entries[tmp_blk.blknum:])

    def build(self, data_list, pages_per_range, is_sorted, region, data_precision):
        # 1. save xy index
        if is_sorted:
            self.index_entries = data_list.tolist()
            geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        else:
            self.index_entries = [tuple(data) for data in data_list.tolist()]
            geohash = None
        # 2. create meta
        self.meta = Meta(1, pages_per_range, 0, region.right - region.left, region.up - region.bottom,
                         is_sorted, geohash)
        # 3. create blk by data
        blk_size = len(data_list) // self.meta.datas_per_range
        self.block_ranges = [BlockRange(i * self.meta.datas_per_range,
                                        get_mbr_by_points(data_list[i * self.meta.datas_per_range:
                                                                    (i + 1) * self.meta.datas_per_range]))
                             for i in range(blk_size)]
        # 4. create tmp blk
        self.create_tmp_blk()
        self.sum_up_tmp_blk()

    def point_query_blk(self, point):
        """
        找到可能包含xy的blk
        """
        return [blk
                for blk in self.block_ranges
                if blk.value[0] <= point[1] <= blk.value[1] and blk.value[2] <= point[0] <= blk.value[3]]

    def range_query_blk(self, window):
        """
        找到可能和window相交的blk及其空间关系(相交=1/window包含value=2)
        包含关系可以加速查询，即包含意味着blk内所有数据都符合条件
        """
        return [[blk, intersect(window, blk.value)]
                for blk in self.block_ranges]

    def binary_search_duplicate(self, x, left, right):
        """
        binary search geohash in ies[left, right]
        """
        result = []
        ie_max_key = len(self.index_entries) - 1
        if right > ie_max_key:
            right = ie_max_key
        while left <= right:
            mid = (left + right) // 2
            if self.index_entries[mid][2] == x:
                result.append(self.index_entries[mid][-1])
                mid_left = mid - 1
                while mid_left >= left and self.index_entries[mid_left][2] == x:
                    result.append(self.index_entries[mid_left][-1])
                    mid_left -= 1
                mid_right = mid + 1
                while mid_right <= right and self.index_entries[mid_right][2] == x:
                    result.append(self.index_entries[mid_right][-1])
                    mid_right += 1
                return result
            elif self.index_entries[mid][2] < x:
                left = mid + 1
            else:
                right = mid - 1
        return result

    def point_query_single(self, point):
        """
        1. 根据xy找到可能存在的blks
        2. 精确过滤blks对应磁盘范围内的数据
        """
        # 1. 根据xy找到可能存在的blks
        blks = self.point_query_blk(point)
        # 2. 精确过滤blks对应磁盘范围内的数据
        if self.meta.is_sorted:
            gh = self.meta.geohash.encode(point[0], point[1])
            result = []
            for blk in blks:
                self.io_cost += self.meta.pages_per_range
                result.extend(self.binary_search_duplicate(gh, blk.blknum, blk.blknum + self.meta.datas_per_range - 1))
            return result
        else:
            self.io_cost += math.ceil(len(blks) * self.meta.pages_per_range)
            return [ie[-1]
                    for blk in blks
                    for ie in self.index_entries[blk.blknum: blk.blknum + self.meta.datas_per_range]
                    if ie[0] == point[0] and ie[1] == point[1]]

    def range_query_single(self, window):
        """
        1. 根据window找到相交和包含的blks
        2. 精确过滤相交的blks对应磁盘范围内的数据
        3. 直接添加包含的blks对应磁盘范围内的数据
        """
        # 1. 根据window找到相交和包含的blks
        target_blks = self.range_query_blk(window)
        result = []
        for target_blk in target_blks:
            if target_blk[1] == 0:
                continue
            # 3. 直接添加包含的blks对应磁盘范围内的数据
            elif target_blk[1] == 2:
                blk = target_blk[0]
                self.io_cost += self.meta.pages_per_range
                result.extend([ie[-1]
                               for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]])
            # 2. 精确过滤相交的blks对应磁盘范围内的数据
            else:
                blk = target_blk[0]
                self.io_cost += self.meta.pages_per_range
                result.extend([ie[-1]
                               for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
        return result

    def knn_query_single(self, knn):
        """
        1. init window
        2. iter: query target points by range query
        3. if target points is not enough, set window = 2 * window
        4. elif target points is enough, but some target points is in the corner, set window = dst
        """
        # 1. init window
        x, y, k = knn
        k = int(k)
        window_ratio = (k / len(self.index_entries)) ** 0.5
        window_radius = window_ratio * self.meta.region_width / 2
        tp_list = []
        old_window = None
        while True:
            window = [y - window_radius, y + window_radius, x - window_radius, x + window_radius]
            # 2. iter: query target points by range query
            target_blks = self.range_query_blk(window)
            tmp_tp_list = []
            if old_window:
                for target_blk in target_blks:
                    if target_blk[1] == 0:
                        continue
                    elif target_blk[1] == 2:
                        blk = target_blk[0]
                        self.io_cost += self.meta.pages_per_range
                        tmp_tp_list.extend(
                            [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[-1]]
                             for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]
                             if not (old_window[0] <= ie[1] <= old_window[1] and
                                     old_window[2] <= ie[0] <= old_window[3])])
                    else:
                        blk = target_blk[0]
                        self.io_cost += self.meta.pages_per_range
                        tmp_tp_list.extend(
                            [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[-1]]
                             for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]
                             if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3] and
                             not (old_window[0] <= ie[1] <= old_window[1] and old_window[2] <= ie[0] <= old_window[3])])
            else:
                for target_blk in target_blks:
                    if target_blk[1] == 0:
                        continue
                    elif target_blk[1] == 2:
                        blk = target_blk[0]
                        self.io_cost += self.meta.pages_per_range
                        tmp_tp_list.extend(
                            [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[-1]]
                             for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]])
                    else:
                        blk = target_blk[0]
                        self.io_cost += self.meta.pages_per_range
                        tmp_tp_list.extend(
                            [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[-1]]
                             for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]
                             if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
            tp_list.extend(tmp_tp_list)
            old_window = window
            # 3. if target points is not enough, set window = 2 * window
            if len(tp_list) < k:
                window_radius *= 2
            else:
                # 4. elif target points is enough, but some target points is in the corner, set window = dst
                if len(tmp_tp_list):
                    tp_list.sort()
                    dst = tp_list[k - 1][0] ** 0.5
                    if dst > window_radius:
                        window_radius = dst
                    else:
                        break
                else:
                    break
        return [tp[1] for tp in tp_list[:k]]

    def save(self):
        brins_meta = [self.meta.version, self.meta.pages_per_range, self.meta.last_revmap_page,
                      self.meta.region_width, self.meta.region_height]
        if self.meta.is_sorted:
            brins_meta.extend([1, self.meta.geohash.data_precision,
                               self.meta.geohash.region.bottom, self.meta.geohash.region.up,
                               self.meta.geohash.region.left, self.meta.geohash.region.right])
            index_entries = np.array(self.index_entries,
                                     dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')])
        else:
            brins_meta.extend([0, 0, 0, 0, 0, 0])
            index_entries = np.array(self.index_entries,
                                     dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i4'), ("3", 'i4')])
        brins_meta = np.array(tuple(brins_meta))
        brins_blk = [(blk.blknum, blk.value[0], blk.value[1], blk.value[2], blk.value[3]) for blk in self.block_ranges]
        brins_blk = np.array(brins_blk, dtype=[("0", 'i4'), ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8')])
        np.save(os.path.join(self.model_path, 'brins_meta.npy'), brins_meta)
        np.save(os.path.join(self.model_path, 'brins_blk.npy'), brins_blk)
        np.save(os.path.join(self.model_path, 'index_entries.npy'), index_entries)

    def load(self):
        brins_meta = np.load(os.path.join(self.model_path, 'brins_meta.npy'))
        brins_blk = np.load(os.path.join(self.model_path, 'brins_blk.npy'), allow_pickle=True)
        index_entries = np.load(os.path.join(self.model_path, 'index_entries.npy'), allow_pickle=True)
        if brins_meta[5]:
            is_sorted = True
            geohash = Geohash.init_by_precision(data_precision=brins_meta[6],
                                                region=Region(brins_meta[7], brins_meta[8],
                                                              brins_meta[9], brins_meta[10]))
        else:
            is_sorted = False
            geohash = None
        self.meta = Meta(int(brins_meta[0]), int(brins_meta[1]), int(brins_meta[2]),
                         brins_meta[3], brins_meta[4], is_sorted, geohash)
        self.block_ranges = [BlockRange(blk[0], [blk[1], blk[2], blk[3], blk[4]]) for blk in brins_blk]
        self.index_entries = index_entries.tolist()

    def size(self):
        """
        structure_size = brins_meta.npy + brins_blk.npy
        ie_size = index_entries.npy
        """
        # 实际上：
        # meta一致为os.path.getsize(os.path.join(self.model_path, "brins_meta.npy"))-128=4*11=44
        # blk一致为os.path.getsize(os.path.join(self.model_path, "brins_blk.npy"))-128-64=blk_size*(8*4+4)=blk_size*36
        # revmap为none
        # index_entries一致为os.path.getsize(os.path.join(self.model_path, "index_entries.npy"))-128=data_len*(8*2+4)=data_len*20
        # 理论上：
        # revmap存blk id/pointer=meta.size*(2+4)=meta.size*6
        blk_size = len(self.block_ranges)
        return 44 + \
               blk_size * 36 + \
               blk_size * 6, os.path.getsize(os.path.join(self.model_path, "index_entries.npy")) - 128


class Meta:
    def __init__(self, version, pages_per_range, last_revmap_page, region_width, region_height, is_sorted, geohash):
        # BRIN
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page
        # For compute
        self.datas_per_range = int(PAGE_SIZE / 20) * pages_per_range
        self.datas_per_page = int(PAGE_SIZE / 20)
        # For KNN
        self.region_width = region_width
        self.region_height = region_height
        # For sort
        self.is_sorted = is_sorted
        self.geohash = geohash


class BlockRange:
    def __init__(self, blknum, value):
        # BRIN
        # for compute, blknum = id, not page num
        self.blknum = blknum
        self.value = value


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/brinspatial_10w/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = BRINSpatial(model_path=model_path)
    index_name = index.name
    load_index_from_file = False
    if load_index_from_file:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        # build_data_list = load_data(Distribution.NYCT_10W, 0)
        build_data_list = load_data(data_distribution, 0)
        # 按照pagesize=4096, read_ahead=256, size(pointer)=4, size(x/y)=8, brin整体连续存, meta一个page, blk分页存
        # blk体积=blknum/value=4+4*8=36，一个page存113个blk
        # revmap体积=blkid+blk指针=2+4=6，一个page存682个blk
        # data体积=x/y/key=8*2+4=20，一个page存204个data
        # 10w数据，[5]参数下：大约有10w/5/204=99blk
        # 1meta page，99/113=1regular page，99/682=1revmap page，10w/204=491data page
        # 单次扫描IO为读取brin+读取blk对应ie=1+0
        # 索引体积=xy索引+meta+blk+revmap
        index.build(data_list=build_data_list,
                    pages_per_range=5,
                    is_sorted=True,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution])
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    io_cost = 0
    path = '../../data/query/point_query_nyct.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    range_query_list = load_query(data_distribution, 1).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(range_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    knn_query_list = load_query(data_distribution, 2).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(knn_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Update time: %s" % (end_time - start_time))
    logging.info("Update io cost: %s" % (index.io_cost - io_cost))
    io_cost = index.io_cost
    point_query_list = load_query(data_distribution, 0).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result1.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
