import logging
import os
import time

import numpy as np

from src.spatial_index.common_utils import Region, get_mbr_by_points
from src.spatial_index.spatial_index import SpatialIndex

"""
前提条件:
1. tmp br永远有一个，新数据插入只会插入tmp br，tmp br满了就转为br同时创建新的tmp br
"""

PAGE_SZIE = 4096


class BRINSpatial(SpatialIndex):
    def __init__(self, model_path=None, meta=None, block_ranges=None):
        super(BRINSpatial, self).__init__("BRIN Spatial")
        self.xy_index = None
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
        # datas_per_range: 优化计算所需：每个br的数据容量
        # datas_per_page: 优化计算所需：每个page的数据容量
        self.meta = meta
        # revmap pages由多个revmap分页组成
        # 忽略revmap，pages找blk range的过程，通过br的id和pagesperrange直接完成
        # self.revmaps = revmaps
        # regular pages由多个block range分页组成
        # blknum: pages偏移 = id * pagesperrange
        # value: 改动：br的MBR
        self.block_ranges = block_ranges

    def insert_single(self, point):
        # 1. append in xy index
        self.xy_index.append(point)
        # 2. create tmp br if point is on the breakpoint
        if point[-1] % self.meta.datas_per_range == 0:
            self.block_ranges[-1].value = get_mbr_by_points(self.xy_index[-self.meta.datas_per_range:])
            self.create_tmp_br()

    def create_tmp_br(self):
        self.block_ranges.append(BlockRange(blknum=len(self.block_ranges) * self.meta.pages_per_range, value=None))

    def build(self, data_list, pages_per_range):
        # 1. save xy index
        data_list = [tuple(data) for data in data_list.tolist()]
        self.xy_index = data_list
        # 2. create meta
        self.meta = Meta(1, pages_per_range, 0)
        # 3. create br by data
        br_size = len(data_list) // self.meta.datas_per_range
        self.block_ranges = [BlockRange(i * pages_per_range,
                                        get_mbr_by_points(data_list[i * self.meta.datas_per_range:
                                                                    (i + 1) * self.meta.datas_per_range]))
                             for i in range(br_size)]
        # 4. create tmp br
        self.create_tmp_br()

    def point_query_br(self, point):
        """
        找到可能包含xy的br的key，不包含tmp br
        """
        return [br
                for br in self.block_ranges[:-1]
                if br.value.contain_and_border_by_list(point)]

    def range_query_br(self, window):
        """
        找到可能和window相交的br的key及其空间关系(相交=1/window包含value=2)，不包含tmp br
        包含关系可以加速查询，即包含意味着br内所有数据都符合条件
        """
        return [[br, Region.init_by_list(window).intersect(br.value)]
                for br in self.block_ranges[:-1]
                if br.value is not None]

    def point_query_single(self, point):
        """
        1. 根据xy找到可能存在的brs
        2. 精确过滤brs和tmp br对应磁盘范围内的数据
        """
        # 1. 根据xy找到可能存在的brs
        brs = self.point_query_br(point)
        # 2. 精确过滤tmp br对应磁盘范围内的数据
        result = [xy[2]
                  for xy in self.xy_index[self.block_ranges[-1].blknum * self.meta.datas_per_page:]
                  if xy[0] == point[0] and xy[1] == point[1]]
        # 2. 精确过滤brs对应磁盘范围内的数据
        result.extend([xy[2]
                       for br in brs
                       for xy in self.xy_index[br.blknum * self.meta.datas_per_page:
                                               (br.blknum + self.meta.pages_per_range) * self.meta.datas_per_page]
                       if xy[0] == point[0] and xy[1] == point[1]])
        return result

    def range_query_single(self, window):
        """
        1. 根据window找到相交和包含的brs
        2. 精确过滤相交的brs和tmp br对应磁盘范围内的数据
        3. 直接添加包含的brs对应磁盘范围内的数据
        """
        # 1. 根据window找到相交和包含的brs
        target_brs = self.range_query_br(window)
        # 2. 精确过滤相交的tmp br对应磁盘范围内的数据
        result = [xy[2] for xy in self.xy_index[self.block_ranges[-1].blknum * self.meta.datas_per_page:]
                  if window[0] <= xy[0] <= window[1] and window[2] <= xy[1] <= window[3]]
        for target_br in target_brs:
            if target_br[1] == 0:
                continue
            # 3. 直接添加包含的brs对应磁盘范围内的数据
            elif target_br[1] == 2:
                br = target_br[0]
                result.extend(self.xy_index[br.blknum * self.meta.datas_per_page:
                                            (br.blknum + + self.meta.pages_per_range) * self.meta.datas_per_page])
            # 2. 精确过滤相交的brs对应磁盘范围内的数据
            else:
                br = target_br[0]
                result.extend([xy[2] for xy in self.xy_index[
                                               br.blknum * self.meta.datas_per_page:
                                               (br.blknum + + self.meta.pages_per_range) * self.meta.datas_per_page]
                               if window[0] <= xy[1] <= window[1] and window[2] <= xy[0] <= window[3]])
        return result

    def save(self):
        brins_meta = np.array(
            (self.meta.version, self.meta.pages_per_range, self.meta.last_revmap_page))
        brins_br = [(br.blknum, br.value.bottom, br.value.up, br.value.left, br.value.right) for br in
                    self.block_ranges[:-1]]
        br = self.block_ranges[-1]
        if br.value is None:
            brins_br.append((br.blknum, -1, -1, -1, -1))
        else:
            brins_br.append((br.blknum, br.value.bottom, br.value.up, br.value.left, br.value.right))
        brins_br = np.array(brins_br, dtype=[("0", 'i4'), ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8')])
        np.save(os.path.join(self.model_path, 'brins_meta.npy'), brins_meta)
        np.save(os.path.join(self.model_path, 'brins_br.npy'), brins_br)
        xy_index = np.array(self.xy_index, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i4')])
        np.save(os.path.join(self.model_path, 'xy_index.npy'), xy_index)

    def load(self):
        brins_meta = np.load(self.model_path + 'brins_meta.npy')
        brins_br = np.load(self.model_path + 'brins_br.npy', allow_pickle=True)
        xy_index = np.load(self.model_path + 'xy_index.npy', allow_pickle=True)
        self.meta = Meta(brins_meta[0], brins_meta[1], brins_meta[2])
        brs = []
        for i in range(len(brins_br)):
            br = brins_br[i]
            if br[1] == -1:
                region = None
            else:
                region = Region(br[1], br[2], br[3], br[4])
            brs.append(BlockRange(br[0], region))
        self.block_ranges = brs
        self.xy_index = xy_index.tolist()

    def size(self):
        """
        size = brins_meta.npy + brins_br.npy + xy_index.npy
        """
        # 实际上：
        # meta一致为为os.path.getsize(os.path.join(self.model_path, "brins_meta.npy"))-128=4*3=12
        # br一致为os.path.getsize(os.path.join(self.model_path, "brins_br.npy"))-128-64=br_size*(8*4+4)=br_size*36
        # revmap为none
        # xy_index一致为os.path.getsize(os.path.join(self.model_path, "xy_index.npy"))-128=data_len*(8*2+4)=data_len*20
        # 理论上：
        # revmap存br id/pointer=meta.size*(4+4)=meta.size*8
        br_size = len(self.block_ranges)
        return 12 + \
               br_size * 36 + \
               br_size * 8 + \
               os.path.getsize(os.path.join(self.model_path, "xy_index.npy")) - 128


class Meta:
    def __init__(self, version, pages_per_range, last_revmap_page):
        # BRIN
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page
        # For compute
        self.datas_per_range = int(PAGE_SZIE / 20) * pages_per_range
        self.datas_per_page = int(PAGE_SZIE / 20)


class BlockRange:
    def __init__(self, blknum, value):
        # BRIN
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
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)[:, [10, 11, -1]]
        # 按照pagesize=4096, prefetch=256, size(pointer)=4, size(x/y)=8, brin整体连续存, meta一个page, br分页存
        # br体积=blknum/value=4+4*8=36，一个page存114个br
        # revmap体积=brid+br指针=4+4=8，一个page存512个br
        # data体积=x/y/key=8*2+4=20，一个page存204个data
        # 10w数据，[5]参数下：大约有10w/5/204=99br
        # 1meta page，99/114=1regular page，99/512=1revmap page，10w/204=491data page
        # 单次扫描IO为读取brin+读取br对应xy数据=1+0
        # 索引体积=xy索引+meta+br+revmap
        index.build(data_list=data_list,
                    pages_per_range=5)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    logging.info("Index size: %s" % index.size())
    path = '../../data/query/point_query_10w.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/range_query_10w.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time:  %s" % search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/table/trip_data_2_filter_10w.npy'
    insert_data_list = np.load(path, allow_pickle=True)[:2000, [10, 11, -1]]
    index.insert(insert_data_list)
    start_time = time.time()
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
