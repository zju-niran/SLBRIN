import logging
import math
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import get_mbr_by_points, contain_and_border, intersect
from src.spatial_index.spatial_index import SpatialIndex

"""
前提条件:
1. tmp blk永远有一个，新数据插入只会插入tmp blk，tmp blk满了就转为blk同时创建新的tmp blk
"""

RA_PAGES = 256
PAGE_SIZE = 4096
RANGE_SIZE = 8 * 4 + 4  # 36
REVMAP_SIZE = 4 + 2  # 6
ITEM_SIZE = 8 * 2 + 4  # 20


class BRINSpatial(SpatialIndex):
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
        self.meta = meta
        # revmap pages由多个revmap分页组成
        # 忽略revmap，pages找block的过程，通过blk的id和pagesperrange直接完成
        # self.revmaps = revmaps
        # regular pages由多个block分页组成
        # blknum: pages偏移 = id * pagesperrange，为便于检索，直接存id
        # value: 改动：blk的MBR
        self.block_ranges = block_ranges

    def insert_single(self, point):
        # 1. append in xy index
        self.index_entries.append(point)
        # 2. create tmp blk if point is on the breakpoint
        if point[-1] % self.meta.datas_per_range == 0:
            self.block_ranges[-1].value = get_mbr_by_points(self.index_entries[-self.meta.datas_per_range:])
            self.create_tmp_blk()

    def create_tmp_blk(self):
        self.block_ranges.append(BlockRange(blknum=len(self.block_ranges) * self.meta.datas_per_range, value=None))

    def build(self, data_list, pages_per_range):
        # 1. save xy index
        data_list = [tuple(data) for data in data_list.tolist()]
        self.index_entries = data_list
        # 2. create meta
        self.meta = Meta(1, pages_per_range, 0)
        # 3. create blk by data
        blk_size = len(data_list) // self.meta.datas_per_range
        self.block_ranges = [BlockRange(i * self.meta.datas_per_range,
                                        get_mbr_by_points(data_list[i * self.meta.datas_per_range:
                                                                    (i + 1) * self.meta.datas_per_range]))
                             for i in range(blk_size)]
        # 4. create tmp blk
        self.create_tmp_blk()

    def point_query_blk(self, point):
        """
        找到可能包含xy的blk的key，不包含tmp blk
        """
        return [blk
                for blk in self.block_ranges[:-1]
                if contain_and_border(blk.value, point)]

    def range_query_blk(self, window):
        """
        找到可能和window相交的blk的key及其空间关系(相交=1/window包含value=2)，不包含tmp blk
        包含关系可以加速查询，即包含意味着blk内所有数据都符合条件
        """
        return [[blk, intersect(window, blk.value)]
                for blk in self.block_ranges[:-1]]

    def point_query_single(self, point):
        """
        1. 根据xy找到可能存在的blks
        2. 精确过滤blks和tmp blk对应磁盘范围内的数据
        """
        # 1. 根据xy找到可能存在的blks
        blks = self.point_query_blk(point)
        # 2. 精确过滤tmp blk对应磁盘范围内的数据
        result = [ie[2]
                  for ie in self.index_entries[self.block_ranges[-1].blknum:]
                  if ie[0] == point[0] and ie[1] == point[1]]
        # 2. 精确过滤blks对应磁盘范围内的数据
        result.extend([ie[2]
                       for blk in blks
                       for ie in self.index_entries[blk.blknum: blk.blknum + self.meta.datas_per_range]
                       if ie[0] == point[0] and ie[1] == point[1]])
        return result

    def range_query_single(self, window):
        """
        1. 根据window找到相交和包含的blks
        2. 精确过滤相交的blks和tmp blk对应磁盘范围内的数据
        3. 直接添加包含的blks对应磁盘范围内的数据
        """
        # 1. 根据window找到相交和包含的blks
        target_blks = self.range_query_blk(window)
        # 2. 精确过滤相交的tmp blk对应磁盘范围内的数据
        result = [ie[2]
                  for ie in self.index_entries[self.block_ranges[-1].blknum:]
                  if contain_and_border(window, ie)]
        for target_blk in target_blks:
            if target_blk[1] == 0:
                continue
            # 3. 直接添加包含的blks对应磁盘范围内的数据
            elif target_blk[1] == 2:
                blk = target_blk[0]
                result.extend([ie[2]
                               for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]])
            # 2. 精确过滤相交的blks对应磁盘范围内的数据
            else:
                blk = target_blk[0]
                result.extend([ie[2]
                               for ie in self.index_entries[blk.blknum:blk.blknum + self.meta.datas_per_range]
                               if contain_and_border(window, ie)])
        return result

    def save(self):
        brins_meta = np.array(
            (self.meta.version, self.meta.pages_per_range, self.meta.last_revmap_page))
        brins_blk = [(blk.blknum, blk.value[0], blk.value[1], blk.value[2], blk.value[3]) for blk in
                     self.block_ranges[:-1]]
        blk = self.block_ranges[-1]
        if blk.value is None:
            brins_blk.append((blk.blknum, -1, -1, -1, -1))
        else:
            brins_blk.append((blk.blknum, blk.value[0], blk.value[1], blk.value[2], blk.value[3]))
        brins_blk = np.array(brins_blk, dtype=[("0", 'i4'), ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8')])
        np.save(os.path.join(self.model_path, 'brins_meta.npy'), brins_meta)
        np.save(os.path.join(self.model_path, 'brins_blk.npy'), brins_blk)
        index_entries = np.array(self.index_entries, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i4')])
        np.save(os.path.join(self.model_path, 'index_entries.npy'), index_entries)

    def load(self):
        brins_meta = np.load(os.path.join(self.model_path, 'brins_meta.npy'))
        brins_blk = np.load(os.path.join(self.model_path, 'brins_blk.npy'), allow_pickle=True)
        index_entries = np.load(os.path.join(self.model_path, 'index_entries.npy'), allow_pickle=True)
        self.meta = Meta(brins_meta[0], brins_meta[1], brins_meta[2])
        blks = []
        for i in range(len(brins_blk)):
            blk = brins_blk[i]
            if blk[1] == -1:
                region = None
            else:
                region = [blk[1], blk[2], blk[3], blk[4]]
            blks.append(BlockRange(blk[0], region))
        self.block_ranges = blks
        self.index_entries = index_entries.tolist()

    def size(self):
        """
        structure_size = brins_meta.npy + brins_blk.npy
        ie_size = index_entries.npy
        """
        # 实际上：
        # meta一致为为os.path.getsize(os.path.join(self.model_path, "brins_meta.npy"))-128=4*3=12
        # blk一致为os.path.getsize(os.path.join(self.model_path, "brins_blk.npy"))-128-64=blk_size*(8*4+4)=blk_size*36
        # revmap为none
        # index_entries一致为os.path.getsize(os.path.join(self.model_path, "index_entries.npy"))-128=data_len*(8*2+4)=data_len*20
        # 理论上：
        # revmap存blk id/pointer=meta.size*(2+4)=meta.size*6
        blk_size = len(self.block_ranges)
        return 12 + \
               blk_size * 36 + \
               blk_size * 6, os.path.getsize(os.path.join(self.model_path, "index_entries.npy")) - 128

    def io(self):
        """
        io=获取brin的io+获取data
        """
        range_len = len(self.block_ranges)
        meta_page_len = 1
        regular_page_len = math.ceil(range_len * RANGE_SIZE / PAGE_SIZE)
        revmap_page_len = math.ceil(range_len * REVMAP_SIZE / PAGE_SIZE)
        # io when load brin
        brin_io = math.ceil((meta_page_len + regular_page_len + revmap_page_len) / RA_PAGES)
        # io when load data
        data_io = math.ceil(self.meta.pages_per_range / RA_PAGES)
        return brin_io + data_io


class Meta:
    def __init__(self, version, pages_per_range, last_revmap_page):
        # BRIN
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page
        # For compute
        self.datas_per_range = int(PAGE_SIZE / 20) * pages_per_range
        self.datas_per_page = int(PAGE_SIZE / 20)


class BlockRange:
    def __init__(self, blknum, value):
        # BRIN
        # for compute, blknum = id, not page num
        self.blknum = blknum
        self.value = value


# @profile(precision=8)
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/table/trip_data_1_filter_10w.npy'
    model_path = "model/brinspatial_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = BRINSpatial(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)[:, [10, 11, -1]]
        # 按照pagesize=4096, read_ahead=256, size(pointer)=4, size(x/y)=8, brin整体连续存, meta一个page, blk分页存
        # blk体积=blknum/value=4+4*8=36，一个page存113个blk
        # revmap体积=blkid+blk指针=2+4=6，一个page存682个blk
        # data体积=x/y/key=8*2+4=20，一个page存204个data
        # 10w数据，[5]参数下：大约有10w/5/204=99blk
        # 1meta page，99/113=1regular page，99/682=1revmap page，10w/204=491data page
        # 单次扫描IO为读取brin+读取blk对应ie=1+0
        # 索引体积=xy索引+meta+blk+revmap
        index.build(data_list=data_list,
                    pages_per_range=5)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    logging.info("IO cost: %s" % index.io())
    path = '../../data/query/point_query_nyct.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/range_query_nyct.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/table/trip_data_2_filter_10w.npy'
    insert_data_list = np.load(path, allow_pickle=True)[:, [10, 11, -1]]
    index.insert(insert_data_list)
    start_time = time.time()
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
