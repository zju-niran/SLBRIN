import gc
import json
import logging
import math
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')

from src.learned_model import TrainedNN
from src.learned_model_simple import TrainedNN as TrainedNN_Simple
from src.spatial_index.common_utils import Region, binary_search_less_max, get_nearest_none, sigmoid, quick_sort, Point, \
    biased_search_almost, biased_search, merge_sorted_array
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex

"""
代码上和论文的diff:
1. 数据和geohash索引没有分开，而是放在data_list里：理论上，索引中找到key后能找到数据磁盘位置，然后直接读取point，但代码实现不了
为了模拟和贴近寻道的效率，索引上直接放了数据；
索引size = 数据+索引的size - 数据的size + sbrin.json的size；
索引构建时间 = 数据geohash编码时间+sbrin构建时间
"""


class SBRIN(SpatialIndex):
    def __init__(self, model_path=None, meta=None, block_ranges=None):
        super(SBRIN, self).__init__("SBRIN")
        self.data_list = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # meta page由meta组成
        # version: 序号偏移
        # pages_per_range: pages偏移 = (itemoffset - 1) * pagesperrange
        # last_revmap_page: 改动：max_length长度的整型geohash
        # threshold_number: 新增：blk range 分裂的数量阈值
        # threshold_length: 新增：blk range 分裂的geohash长度阈值
        # geohash: 新增：max_length = geohash.sum_bits
        # first_tmp_br: 新增：记录第一个tmp blk range的位置
        # last_br: 优化计算所需：first_tmp_br - 1
        # size: 优化计算所需：blk range总数
        self.meta = meta
        # revmap pages由多个revmap分页组成
        # 忽略revmap，pages找blk range的过程，通过itemoffset和pagesperrange直接完成
        # self.revmaps = revmaps
        # regular pages由多个block range分页组成
        # itemoffet: 序号偏移
        # blknum: pages偏移 = (itemoffset - 1) * pagesperrange
        # value: 改动：max_length长度的整型geohash
        # length: 新增：blk range geohash的实际length
        # number: 新增：blk range的数据量
        # model: 新增：learned indices
        # scope: 优化计算所需：BRIN-Spatial有，blk range的scope
        # key: 优化计算所需：blk range的索引key范围=[blknum * block_size, blknum * block_size + number]
        # next_value: 优化计算所需：下一个blk range的value
        self.block_ranges = block_ranges

    @staticmethod
    def init_by_dict(d: dict):
        return SBRIN(model_path=d['model_path'],
                     meta=d['meta'],
                     block_ranges=d['block_ranges'])

    def save_to_dict(self):
        return {
            'name': self.name,
            'meta': self.meta,
            'block_ranges': self.block_ranges,
            'model_path': self.model_path
        }

    def insert(self, point):
        # 1. compute geohash from x/y of point
        point.append(self.meta.geohash.encode(point[0], point[1]))
        # 3. insert into sbrin
        # 3.1 update scope of last tmp br and create new tmp br when last tmp br is full
        if self.block_ranges[-1].number >= self.meta.threshold_number:
            last_tmp_br = self.block_ranges[-1]
            x_min, y_min, _ = self.data_list[last_tmp_br.key[1]]
            x_max = x_min
            y_max = y_min
            for data in self.data_list[last_tmp_br.key[0]:last_tmp_br.key[1]]:
                if y_min > data[1]:
                    y_min = data[1]
                elif y_max < data[1]:
                    y_max = data[1]
                if x_min > data[0]:
                    x_min = data[0]
                elif x_max < data[0]:
                    x_max = data[0]
            last_tmp_br.scope = Region(y_min, y_max, x_min, x_max)
            self.create_tmp_br()
            self.data_list.extend([None] * self.meta.threshold_number)
        # 3.2 update data in last tmp br when last tmp br is not full
        last_tmp_br = self.block_ranges[-1]
        last_tmp_br.number += 1
        last_tmp_br.key[1] += 1
        # 3. append in disk
        self.data_list[last_tmp_br.key[1]] = point
        # 4. update sbrin TODO 改成异步
        data_group_dict = self.merge_tmp_br()

    def update(self, points):
        for point in points:
            self.insert(point)

    def build(self, data_list, threshold_number, data_precision, region, use_threshold, threshold, core,
              train_step, batch_num, learning_rate, retrain_time_limit, thread_pool_size, save_nn, weight):
        """
        构建SBRIN
        1. 数据编码和排序
        2. 创建SBRIN
        2.1. 初始化第一个blk range
        2.2. blk range分裂
        2.3. 存储为SBRIN结构
        2.4. 重构数据
        2.5. 预置tmp blk range
        3. 创建Learned Index
        """
        # 1. 数据编码和排序
        # 2. 创建SBRIN
        start_time = time.time()
        # 2.1. 初始化第一个blk range
        n = len(data_list)
        tmp_stack = [(0, 0, n, (0, n - 1), region)]
        result_list = []
        geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        block_size = 100
        threshold_length = region.get_max_depth_by_region_and_precision(precision=data_precision) * 2
        # 2.2. blk range分裂
        while len(tmp_stack):
            cur = tmp_stack.pop(-1)
            if cur[2] > threshold_number and cur[1] < threshold_length:
                child_region = cur[4].split()
                l_key = cur[3][0]
                r_key = cur[3][1]
                tmp_l_key = l_key
                child_list = [None] * 4
                length = cur[1] + 2
                r_bound = cur[0] + (0 << geohash.sum_bits - length)
                for i in range(4):
                    value = r_bound
                    r_bound = cur[0] + (i + 1 << geohash.sum_bits - length)
                    tmp_r_key = binary_search_less_max(data_list, 2, r_bound, tmp_l_key, r_key)
                    child_list[i] = (value, length, tmp_r_key - tmp_l_key + 1, (tmp_l_key, tmp_r_key), child_region[i])
                    tmp_l_key = tmp_r_key + 1
                tmp_stack.extend(child_list[::-1])  # 倒着放入init中，保持顺序
            else:
                # 把不超过的blk range加入结果list，加入的时候顺序为[左上，右下，左上，右上]的逆序，因为堆栈
                result_list.append(cur)
        # 2.3. 存储为SBRIN结构
        pages_per_range = threshold_number // block_size
        # last_revmap_page理论上是第一个regular_page磁盘位置-1
        result_len = len(result_list)
        self.meta = Meta(1, pages_per_range, 0, threshold_number, threshold_length, geohash,
                         max([result[1] for result in result_list]), result_len, result_len - 1, result_len)
        self.block_ranges = [
            BlockRange(i + 1, i * pages_per_range, result_list[i][0], result_list[i][1], result_list[i][2], None,
                       Region.up_right_less_region(result_list[i][4], pow(10, -data_precision - 1)), result_list[i][3],
                       None) for i in range(result_len)]
        for i in range(result_len - 1):
            self.block_ranges[i].next_value = result_list[i + 1][0]
        self.block_ranges[-1].next_value = 1 << self.meta.geohash.sum_bits
        # 2.5. 预置tmp blk range
        self.create_tmp_br()
        # revmaps理论上要记录每个regular的磁盘位置
        # 2.6. 重构数据
        result_data_list = []
        for br in self.block_ranges:
            result_data_list.extend(data_list[br.key[0]: br.key[1] + 1])
            result_data_list.extend([None] * (self.meta.threshold_number - br.number))
            br_first_key = self.meta.threshold_number * (br.itemoffset - 1)
            br.key = [br_first_key, br_first_key + br.number - 1]
        self.data_list = result_data_list
        end_time = time.time()
        self.logging.info("Create SBRIN: %s" % (end_time - start_time))
        # 3. 创建Learned Index
        start_time = time.time()
        self.build_nn_multiprocess(use_threshold, threshold, core, train_step, batch_num, learning_rate,
                                   retrain_time_limit, thread_pool_size, save_nn, weight)
        end_time = time.time()
        self.logging.info("Create learned model: %s" % (end_time - start_time))

    def build_nn_multiprocess(self, use_threshold, threshold, core, train_step, batch_num, learning_rate,
                              retrain_time_limit, thread_pool_size, save_nn, weight):
        multiprocessing.set_start_method('spawn', force=True)  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()
        for i in range(self.meta.first_tmp_br):
            key_bound = self.block_ranges[i].key
            # 训练数据为左下角点+分区数据+右上角点
            inputs = [j[2] for j in self.data_list[key_bound[0]:key_bound[1] + 1]]
            inputs.insert(0, self.block_ranges[i].value)
            inputs.append(self.block_ranges[i].next_value)
            data_num = self.block_ranges[i].number + 2
            labels = list(range(data_num))
            batch_size = 2 ** math.ceil(math.log(data_num / batch_num, 2))
            if batch_size < 1:
                batch_size = 1
            # batch_size = batch_num
            pool.apply_async(self.build_nn,
                             (i, inputs, labels, use_threshold, threshold, core, train_step, batch_size, learning_rate,
                              retrain_time_limit, save_nn, weight, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.block_ranges[key].model = value

    def build_nn(self, model_key, inputs, labels, use_threshold, threshold, core, train_step, batch_size,
                 learning_rate, retrain_time_limit, save_nn, weight, tmp_dict=None):
        # train model
        if save_nn is False:
            tmp_index = TrainedNN_Simple(self.model_path, model_key, inputs, labels, core, train_step, batch_size,
                                         learning_rate, weight)
            tmp_index.train()
        else:
            tmp_index = TrainedNN(self.model_path, str(model_key), inputs, labels, use_threshold, threshold, core,
                                  train_step, batch_size, learning_rate, retrain_time_limit, weight)
            tmp_index.train()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.weights, len(core) - 2,
                                    math.floor(tmp_index.min_err), math.ceil(tmp_index.max_err))
        del tmp_index
        gc.collect()
        tmp_dict[model_key] = abstract_index

    def reconstruct_data_old(self):
        """
        把数据存在predict的地方，如果pre已有数据：
        1. pre处数据的geohash==数据本身的geohash，说明数据重复，则找到离pre最近的[pre-maxerr, pre-minerr]范围内的None来存储
        2. pre处数据的geohash!=数据本身的geohash，说明本该属于数据的位置被其他数据占用了，为了保持有序，找None的过程只往一边走
        存在问题：这种重构相当于在存储数据的时候依旧保持数据分布的稀疏性，但是密集的地方后续往往更加密集，导致这些地方的数据存储位置更加紧张
        这个问题往往在大数据量或误差大或分布不均匀的br更容易出现，即最后"超出边界"的报错
        """
        result_data_list = [None] * self.meta.first_tmp_br * self.meta.threshold_number
        for br in self.block_ranges:
            br.model.output_min = self.meta.threshold_number * (br.itemoffset - 1)
            br.model.output_max = self.meta.threshold_number * br.itemoffset - 1
            for i in range(br.key[0], br.key[1] + 1):
                pre = round(br.model.predict(self.data_list[i][2]))
                if result_data_list[pre] is None:
                    result_data_list[pre] = self.data_list[i]
                else:
                    # 重复数据处理：写入误差范围内离pre最近的None里
                    if result_data_list[pre][2] == self.data_list[i][2]:
                        l_bound = max(round(pre - br.model.max_err), br.model.output_min)
                        r_bound = min(round(pre - br.model.min_err), br.model.output_max)
                    else:  # 非重复数据，但是整型部分重复，或被重复数据取代了位置
                        if result_data_list[pre][2] > self.data_list[i][2]:
                            l_bound = max(round(pre - br.model.max_err), br.model.output_min)
                            r_bound = pre
                        else:
                            l_bound = pre
                            r_bound = min(round(pre - br.model.min_err), br.model.output_max)
                    key = get_nearest_none(result_data_list, pre, l_bound, r_bound)
                    if key is None:
                        # 超出边界是因为大量的数据相互占用导致误差放大
                        print("超出边界")
                    else:
                        result_data_list[key] = self.data_list[i]
        self.data_list = result_data_list

    def create_tmp_br(self):
        new_tmp_br = self.block_ranges[-1].itemoffset
        self.block_ranges.append(
            BlockRange(itemoffset=new_tmp_br + 1,
                       blknum=new_tmp_br * self.meta.pages_per_range,
                       value=None,
                       length=None,
                       number=0,
                       model=None,
                       scope=None,
                       key=[new_tmp_br * self.meta.threshold_number, new_tmp_br * self.meta.threshold_number - 1],
                       next_value=None))
        self.meta.size += 1

    def delete_tmp_br(self, br_num):
        self.meta.size -= br_num
        del self.block_ranges[self.meta.first_tmp_br:self.meta.first_tmp_br + br_num]

    def merge_tmp_br(self):
        merge_br_num = 1
        if self.meta.size - self.meta.first_tmp_br > merge_br_num:
            old_tmp_brs = self.block_ranges[self.meta.first_tmp_br:self.meta.first_tmp_br + merge_br_num]
            first_tmp_br = self.block_ranges[self.meta.first_tmp_br]
            old_data_size = merge_br_num * self.meta.threshold_number
            old_data = self.data_list[
                       first_tmp_br.key[0]:first_tmp_br.key[0] + merge_br_num * self.meta.threshold_number]
            # tmp br数据排序和分区
            quick_sort(old_data, 2, 0, old_data_size - 1)
            br_key = self.binary_search_less_max(old_data[0][2], 0, self.meta.last_br) + 1
            tmp_l_key = 0
            tmp_r_key = 0
            data_group_dict = {}
            while tmp_r_key < self.meta.threshold_number:
                if self.block_ranges[br_key].value > old_data[tmp_r_key][2]:
                    tmp_r_key += 1
                else:
                    if tmp_r_key - tmp_l_key > 1:
                        data_group_dict[br_key - 1] = old_data[tmp_l_key:tmp_r_key]
                        tmp_l_key = tmp_r_key
                    br_key += 1
            # tmp br合并到br：是否更新/数据合并/模型更新
            for br_key in data_group_dict:
                new_data_list = data_group_dict[br_key]
                new_data_len = len(new_data_list)
                br = self.block_ranges[br_key]
                # 分裂br
                if new_data_len + br.number > self.meta.threshold_number and br.length < self.meta.threshold_length:
                    return
                else:
                    # self.data_list[br.key[1] + 1:br.key[1] + 1 + new_data_len] = new_data_list
                    # quick_sort(self.data_list, 2, br.key[0], br.key[1] + new_data_len)
                    merge_sorted_array(self.data_list, 2, br.key[0], br.key[1], new_data_list)
                    br.number += new_data_len
                    br.key[1] += new_data_len
                    br.model_update_err(self.data_list)
            # 销毁tmp br
            self.delete_tmp_br(br_num=merge_br_num)

            print("")

    def point_query_br(self, point):
        """
        根据geohash找到所在的blk range的key
        1. 计算geohash对应到blk range的geohash_int
        2. 找到比geohash_int小的最大值即为geohash所在的blk range
        """
        return self.binary_search_less_max(point, 0, self.meta.last_br)

    def range_query_br_old(self, point1, point2, window):
        """
        根据geohash1/geohash2找到之间所有blk range以及blk range和window的相交关系
        1. 使用point_query_br查找geohash1/geohash2所在blk range
        2. 返回blk range1和blk range2之间的所有blk range，以及他们和window的的包含关系
        TODO: intersect函数还可以改进，改为能判断window对于region的上下左右关系
        """
        i = self.binary_search_less_max(point1, 0, self.meta.last_br)
        j = self.binary_search_less_max(point2, i, self.meta.last_br)
        if i == j:
            return [((3, None), self.block_ranges[i])]
        else:
            return [(window.intersect(self.block_ranges[k].scope), self.block_ranges[k]) for k in range(i, j - 1)]

    def range_query_br(self, point1, point2):
        """
        根据geohash1/geohash2找到之间所有blk range的key以及和window的位置关系
        1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
        3. 根据tgt_geohash分组并合并position
        """
        # 1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        br_key1 = self.binary_search_less_max(point1, 0, self.meta.last_br)
        br_key2 = self.binary_search_less_max(point2, br_key1, self.meta.last_br)
        if br_key1 == br_key2:
            return {br_key1: 15}
        else:
            org_geohash_list = self.meta.geohash.ranges_by_int(point1, point2, self.meta.max_length)
            # 2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
            # 3. 根据tgt_geohash分组并合并position
            size = len(org_geohash_list) - 1
            i = 1
            tgt_geohash_dict = {br_key1: org_geohash_list[0][1],
                                br_key2: org_geohash_list[-1][1]}
            while i < size:
                if self.block_ranges[br_key1].value > org_geohash_list[i][0]:
                    tgt_geohash_dict[br_key1 - 1] = tgt_geohash_dict.get(br_key1 - 1, 0) | org_geohash_list[i][1]
                    i += 1
                else:
                    br_key1 += 1
            return tgt_geohash_dict
            # 前缀匹配太慢：时间复杂度=O(len(window对应的geohash个数)*(j-i))

    def knn_query_br(self, point1, point2, point3):
        """
        根据geohash1/geohash2找到之间所有blk range的key以及和window的位置关系，并基于和point3距离排序
        1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
        3. 根据tgt_geohash分组并合并position
        4. 计算每个tgt_geohash和point3的距离，并进行降序排序
        """
        # 1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        br_key1 = self.binary_search_less_max(point1, 0, self.meta.last_br)
        br_key2 = self.binary_search_less_max(point2, br_key1, self.meta.last_br)
        if br_key1 == br_key2:
            return [[br_key1, 15, 0]]
        else:
            org_geohash_list = self.meta.geohash.ranges_by_int(point1, point2, self.meta.max_length)
            # 2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
            # 3. 根据tgt_geohash分组并合并position
            size = len(org_geohash_list) - 1
            i = 1
            tgt_geohash_dict = {br_key1: org_geohash_list[0][1],
                                br_key2: org_geohash_list[-1][1]}
            while i < size:
                if self.block_ranges[br_key1].value > org_geohash_list[i][0]:
                    tgt_geohash_dict[br_key1 - 1] = tgt_geohash_dict.get(br_key1 - 1, 0) | org_geohash_list[i][1]
                    i += 1
                else:
                    br_key1 += 1
            # 4. 计算每个tgt_geohash和point3的距离，并进行降序排序
            return sorted([[tgt_geohash,
                            tgt_geohash_dict[tgt_geohash],
                            self.block_ranges[tgt_geohash].scope.get_min_distance_pow_by_point_list(point3)]
                           for tgt_geohash in tgt_geohash_dict], key=lambda x: x[2])

    def binary_search_less_max(self, x, left, right):
        """
        二分查找比x小的最大值
        优化: 循环->二分:15->1
        """
        while left <= right:
            mid = (left + right) // 2
            if self.block_ranges[mid].value == x:
                return mid
            elif self.block_ranges[mid].value < x:
                left = mid + 1
            else:
                right = mid - 1
        return right

    def point_query_single(self, point):
        """
        query key by x/y point
        1. compute geohash from x/y of points
        2. find blk range within geohash by sbrin.point_query
        3. predict by leaf model
        4. biased search in scope [pre - max_err, pre + min_err]
        """
        # 1. compute geohash from x/y of point
        gh = self.meta.geohash.encode(point[0], point[1])
        # 2. find blk range within geohash by sbrin.point_query
        blk_range = self.block_ranges[self.point_query_br(gh)]
        if blk_range.number == 0:
            return None
        else:
            # 3. predict by leaf model
            pre = blk_range.model_predict(gh)
            # 4. biased search in scope [pre - max_err, pre + min_err]
            return biased_search(self.data_list, 2, gh, pre,
                                 max(pre - blk_range.model.max_err, blk_range.key[0]),
                                 min(pre - blk_range.model.min_err, blk_range.key[1]))

    def range_query_single_old(self, window):
        """
        query key by x1/y1/x2/y2 window
        1. compute geohash from window_left and window_right
        2. get all the blk range and its relationship with window between geohash1/geohash2 by sbrin.range_query
        3. for different relation, use different method to handle the points
        3.1 if window contain the blk range, add all the items into results
        3.2 if window intersect or within the blk range
        3.2.1 get the min_geohash/max_geohash of intersect part
        3.2.2 get the min_key/max_key by nn predict and biased search
        3.2.3 filter all the point of scope[min_key/max_key] by range.contain(point)
        主要耗时间：两次geohash, predict和最后的精确过滤，0.1, 0.1 , 0.6
        # TODO: 由于build sbrin的时候region移动了，导致这里的查询不准确了
        """
        region = Region(window[0], window[1], window[2], window[3])
        # 1. compute geohash of window_left and window_right
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        # 2. get all the blk range and its relationship with window between geohash1/geohash2 by sbrin.range_query
        blk_range_list = self.range_query_br_old(gh1, gh2, region)
        result = []
        # 3. for different relation, use different method to handle the points
        for blk_range in blk_range_list:
            # 0 2 1 3的顺序是按照频率降序
            if blk_range[0][0] == 0:  # no relation
                continue
            else:
                if blk_range[1].number == 0:  # blk range is None
                    continue
                # 3.1 if window contain the blk range, add all the items into results
                if blk_range[0][0] == 2:  # window contain blk range
                    result.extend(list(range(blk_range[1].key[0], blk_range[1].key[1] + 1)))
                # 3.2 if window intersect or within the blk range
                else:
                    # 3.2.1 get the min_geohash/max_geohash of intersect part
                    if blk_range[0][0] == 1:  # intersect
                        gh1 = self.meta.geohash.encode(blk_range[0][1].left, blk_range[0][1].bottom)
                        gh2 = self.meta.geohash.encode(blk_range[0][1].right, blk_range[0][1].up)
                    # 3.2.2 get the min_key/max_key by nn predict and biased search
                    pre1 = blk_range[1].model_predict(gh1)
                    pre2 = blk_range[1].model_predict(gh2)
                    min_err = blk_range[1].model.min_err
                    max_err = blk_range[1].model.max_err
                    l_bound1 = max(pre1 - max_err, blk_range[1].key[0])
                    r_bound1 = min(pre1 - min_err, blk_range[1].key[1])
                    key_left = biased_search(self.data_list, 2, gh1, pre1, l_bound1, r_bound1)
                    if gh1 == gh2:
                        if len(key_left) > 0:
                            result.extend(key_left)
                    else:
                        key_left = l_bound1 if len(key_left) == 0 else min(key_left)
                        l_bound2 = max(pre2 - max_err, blk_range[1].key[0])
                        r_bound2 = min(pre2 - min_err, blk_range[1].key[1])
                        key_right = biased_search(self.data_list, 2, gh2, pre2, l_bound2, r_bound2)
                        key_right = r_bound2 if len(key_right) == 0 else max(key_right)
                        # 3.2.3 filter all the point of scope[min_key/max_key] by range.contain(point)
                        result.extend([key for key in range(key_left, key_right + 1)
                                       if region.contain_and_border_by_list(self.data_list[key])])
        return result

    def range_query_old(self, windows):
        return [self.range_query_single_old(window) for window in windows]

    def range_query_single(self, window):
        """
        query key by x1/y1/x2/y2 window
        1. compute geohash from window_left and window_right
        2. get all relative blk ranges with key and relationship
        3. get min_geohash and max_geohash of every blk range for different relation
        4. predict min_key/max_key by nn
        5. filter all the point of scope[min_key/max_key] by range.contain(point)
        主要耗时间：sbrin.range_query.ranges_by_int/nn predict/精确过滤: 307mil/145mil/359mil
        """
        if window[0] == window[1] and window[2] == window[3]:
            return self.point_query_single([window[2], window[0]])
        # 1. compute geohash of window_left and window_right
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        # 2. get all relative blk ranges with key and relationship
        br_list = self.range_query_br(gh1, gh2)
        result = []
        # 3. get min_geohash and max_geohash of every blk range for different relation
        position_func_list = [lambda reg: (None, None, None),
                              lambda reg: (  # right
                                  None,
                                  self.meta.geohash.encode(window[3], reg.up),
                                  lambda x: window[3] >= x[0]),
                              lambda reg: (  # left
                                  self.meta.geohash.encode(window[2], reg.bottom),
                                  None,
                                  lambda x: window[2] <= x[0]),
                              lambda reg: (  # left-right
                                  self.meta.geohash.encode(window[2], reg.bottom),
                                  self.meta.geohash.encode(window[3], reg.up),
                                  lambda x: window[2] <= x[0] <= window[3]),
                              lambda reg: (  # up
                                  None,
                                  self.meta.geohash.encode(reg.right, window[1]),
                                  lambda x: window[1] >= x[1]),
                              lambda reg: (  # up-right
                                  None,
                                  gh2,
                                  lambda x: window[3] >= x[0] and window[1] >= x[1]),
                              lambda reg: (  # up-left
                                  self.meta.geohash.encode(window[2], reg.bottom),
                                  self.meta.geohash.encode(reg.right, window[1]),
                                  lambda x: window[2] <= x[0] and window[1] >= x[1]),
                              lambda reg: (  # up-left-right
                                  self.meta.geohash.encode(window[2], reg.bottom),
                                  gh2,
                                  lambda x: window[2] <= x[0] <= window[3] and window[1] >= x[1]),
                              lambda reg: (  # bottom
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  None,
                                  lambda x: window[0] <= x[1]),
                              lambda reg: (  # bottom-right
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  self.meta.geohash.encode(window[3], reg.up),
                                  lambda x: window[3] >= x[0] and window[0] <= x[1]),
                              lambda reg: (  # bottom-left
                                  gh1,
                                  None,
                                  lambda x: window[2] <= x[0] and window[0] <= x[1]),
                              lambda reg: (  # bottom-left-right
                                  gh1,
                                  self.meta.geohash.encode(window[3], reg.up),
                                  lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1]),
                              lambda reg: (  # bottom-up
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  self.meta.geohash.encode(reg.right, window[1]),
                                  lambda x: window[0] <= x[1] <= window[1]),
                              lambda reg: (  # bottom-up-right
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  gh2,
                                  lambda x: window[3] >= x[0] and window[0] <= x[1] <= window[1]),
                              lambda reg: (  # bottom-up-left
                                  gh1,
                                  self.meta.geohash.encode(reg.right, window[1]),
                                  lambda x: window[2] <= x[0] and window[0] <= x[1] <= window[1]),
                              lambda reg: (  # bottom-up-left-right
                                  gh1,
                                  gh2,
                                  lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1] <= window[1])]
        for br_key in br_list:
            br = self.block_ranges[br_key]
            if br.number == 0:  # blk range is None
                continue
            position = br_list[br_key]
            if position == 0:  # window contain blk range
                result.extend(list(range(br.key[0], br.key[1] + 1)))
            else:
                # if-elif-else->lambda, 30->4
                gh_new1, gh_new2, compare_func = position_func_list[position](br.scope)
                # 4 predict min_key/max_key by nn
                if gh_new1 is not None:
                    pre1 = br.model_predict(gh_new1)
                    l_bound1 = max(pre1 - br.model.max_err, br.key[0])
                    r_bound1 = min(pre1 - br.model.min_err, br.key[1])
                    key_left = min(biased_search_almost(self.data_list, 2, gh_new1, pre1, l_bound1, r_bound1))
                else:
                    key_left = br.key[0]
                if gh_new2 is not None:
                    pre2 = br.model_predict(gh_new2)
                    l_bound2 = max(pre2 - br.model.max_err, br.key[0])
                    r_bound2 = min(pre2 - br.model.min_err, br.key[1])
                    key_right = max(biased_search_almost(self.data_list, 2, gh_new2, pre2, l_bound2, r_bound2))
                else:
                    key_right = br.key[1]
                # 5 filter all the point of scope[min_key/max_key] by range.contain(point)
                # 优化: region.contain->compare_func不同位置的点做不同的判断: 638->474mil
                result.extend([key for key in range(key_left, key_right + 1)
                               if compare_func(self.data_list[key])])
        return result

    def knn_query_single(self, knn):
        """
        query key by x1/y1/n knn
        1. get the nearest key of query point
        2. get the nn points to create range query window
        3. filter point by distance
        主要耗时间：sbrin.knn_query.ranges_by_int/nn predict/精确过滤: 4.7mil/21mil/14.4mil
        """
        k = knn[2]
        # 1. get the nearest key of query point
        qp_g = self.meta.geohash.encode(knn[0], knn[1])
        qp_blk_key = self.point_query_br(qp_g)
        qp_blk = self.block_ranges[qp_blk_key]
        # if blk range is None, qp_key = the max key of the last blk range
        if qp_blk.number == 0:
            query_point_key = qp_blk.key[1]
        # if model is not None, qp_key = point_query(geohash)
        else:
            pre = qp_blk.model_predict(qp_g)
            l_bound = max(pre - qp_blk.model.max_err, qp_blk.key[0])
            r_bound = min(pre - qp_blk.model.min_err, qp_blk.key[1])
            query_point_key = biased_search_almost(self.data_list, 2, qp_g, pre, l_bound, r_bound)[0]
        # 2. get the n points to create range query window
        # TODO: 两种策略，一种是左右找一半，但是如果跳跃了，window很大；还有一种是两边找n，减少跳跃，使window变小
        tp_list = [[Point.distance_pow_point_list(knn, self.data_list[query_point_key]), query_point_key]]
        cur_key = query_point_key + 1
        cur_block_key = qp_blk_key
        i = 0
        while i < k - 1:
            if self.data_list[cur_key] is None:
                cur_block_key += 1
                if cur_block_key > self.meta.last_br:
                    break
                cur_key = self.block_ranges[cur_block_key].key[0]
            else:
                tp_list.append([Point.distance_pow_point_list(knn, self.data_list[cur_key]), cur_key])
                cur_key += 1
                i += 1
        cur_key = query_point_key - 1
        cur_block_key = qp_blk_key
        i = 0
        while i < k - 1:
            if self.data_list[cur_key] is None:
                cur_block_key -= 1
                if cur_block_key < 0:
                    break
                cur_key = self.block_ranges[qp_blk_key].key[1]
            else:
                tp_list.append([Point.distance_pow_point_list(knn, self.data_list[cur_key]), cur_key])
                cur_key -= 1
                i += 1
        tp_list = sorted(tp_list)[:k]
        max_dist = tp_list[-1][0]
        if max_dist == 0:
            return [tp[1] for tp in tp_list]
        max_dist_pow = max_dist ** 0.5
        window = [knn[1] - max_dist_pow, knn[1] + max_dist_pow, knn[0] - max_dist_pow, knn[0] + max_dist_pow]
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        tp_window_brs = self.knn_query_br(gh1, gh2, knn)
        position_func_list = [lambda reg: (None, None),  # window contain blk range
                              lambda reg: (  # right
                                  None,
                                  self.meta.geohash.encode(window[3], reg.up)),
                              lambda reg: (  # left
                                  self.meta.geohash.encode(window[2], reg.bottom),
                                  None),
                              None,  # left-right
                              lambda reg: (  # up
                                  None,
                                  self.meta.geohash.encode(reg.right, window[1])),
                              lambda reg: (  # up-right
                                  None,
                                  gh2),
                              lambda reg: (  # up-left
                                  self.meta.geohash.encode(window[2], reg.bottom),
                                  self.meta.geohash.encode(reg.right, window[1])),
                              lambda reg: (None, None),  # up-left-right
                              lambda reg: (  # bottom
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  None),
                              lambda reg: (  # bottom-right
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  self.meta.geohash.encode(window[3], reg.up)),
                              lambda reg: (  # bottom-left
                                  gh1,
                                  None),
                              lambda reg: (  # bottom-left-right
                                  gh1,
                                  self.meta.geohash.encode(window[3], reg.up)),
                              None,
                              lambda reg: (  # bottom-up-right
                                  self.meta.geohash.encode(reg.left, window[0]),
                                  gh2),
                              lambda reg: (  # bottom-up-left
                                  gh1,
                                  self.meta.geohash.encode(reg.right, window[1])),
                              lambda reg: (  # bottom-up-left-right
                                  gh1,
                                  gh2)]
        tp_list = []
        for tp_window_blk in tp_window_brs:
            if tp_window_blk[2] > max_dist:
                break
            blk = self.block_ranges[tp_window_blk[0]]
            if blk.number == 0:  # blk range is None
                continue
            blk_key = blk.key
            gh_new1, gh_new2 = position_func_list[tp_window_blk[1]](blk.scope)
            if gh_new1 is not None:
                pre1 = blk.model_predict(gh_new1)
                l_bound1 = max(pre1 - blk.model.max_err, blk_key[0])
                r_bound1 = min(pre1 - blk.model.min_err, blk_key[1])
                key_left = min(biased_search_almost(self.data_list, 2, gh_new1, pre1, l_bound1, r_bound1))
            else:
                key_left = blk_key[0]
            if gh_new2 is not None:
                pre2 = blk.model_predict(gh_new2)
                l_bound2 = max(pre2 - blk.model.max_err, blk_key[0])
                r_bound2 = min(pre2 - blk.model.min_err, blk_key[1])
                key_right = max(biased_search_almost(self.data_list, 2, gh_new2, pre2, l_bound2, r_bound2))

            else:
                key_right = blk_key[1]
            # 3. filter point by distance
            tp_list.extend([[Point.distance_pow_point_list(knn, self.data_list[i]), i]
                            for i in range(key_left, key_right + 1)])
            tp_list = sorted(tp_list)[:k]
            max_dist = tp_list[-1][0]
        return [tp[1] for tp in tp_list]

    def save(self):
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        with open(self.model_path + 'sbrin.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)
        np.save(self.model_path + 'data_list.npy', np.array(self.data_list))

    def load(self):
        with open(self.model_path + 'sbrin.json', "r") as f:
            sbrin = json.load(f, cls=MyDecoder)
            self.meta = sbrin.meta
            self.block_ranges = sbrin.block_ranges
            self.data_list = np.load(self.model_path + 'data_list.npy', allow_pickle=True).tolist()
            del sbrin

    def size(self):
        return os.path.getsize(os.path.join(self.model_path, "sbrin.json")) + os.path.getsize(
            os.path.join(self.model_path, "data_list.npy")) / 3


class Meta:
    def __init__(self, version, pages_per_range, last_revmap_page,
                 threshold_number, threshold_length, geohash, max_length, first_tmp_br, last_br, size):
        # BRIN
        self.version = version
        self.pages_per_range = pages_per_range
        self.last_revmap_page = last_revmap_page
        # SBRIN
        self.threshold_number = threshold_number
        self.threshold_length = threshold_length
        self.geohash = geohash
        self.max_length = max_length
        self.first_tmp_br = first_tmp_br
        # For compute
        self.last_br = last_br
        self.size = size

    @staticmethod
    def init_by_dict(d: dict):
        return Meta(version=d['version'],
                    pages_per_range=d['pages_per_range'],
                    last_revmap_page=d['last_revmap_page'],
                    threshold_number=d['threshold_number'],
                    threshold_length=d['threshold_length'],
                    geohash=d['geohash'],
                    max_length=d['max_length'],
                    first_tmp_br=d['first_tmp_br'],
                    last_br=d['last_br'],
                    size=d['size'])


class BlockRange:
    def __init__(self, itemoffset, blknum, value, length, number, model, scope, key, next_value):
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
        self.key = key
        self.next_value = next_value

    @staticmethod
    def init_by_dict(d: dict):
        return BlockRange(itemoffset=d['itemoffset'],
                          blknum=d['blknum'],
                          value=d['value'],
                          length=d['length'],
                          number=d['number'],
                          model=d['model'],
                          scope=d['scope'],
                          key=d['key'],
                          next_value=d['next_value'])

    def model_predict(self, x):
        x = int(self.model.predict((x - self.value) / (self.next_value - self.value) - 0.5) * self.number)
        if x <= 0:
            return self.key[0]
        elif x >= self.number:
            return self.key[1]
        return self.key[0] + x

    def model_update_err(self, data_list):
        for key in range(self.key[0], self.key[1] + 1):
            y_diff = self.model_predict(data_list[key][2]) - key
            if y_diff > self.model.max_err:
                self.model.max_err = y_diff
            elif y_diff < self.model.min_err:
                self.model.min_err = y_diff


class AbstractNN:
    def __init__(self, weights, hl_nums, min_err, max_err):
        self.weights = weights
        self.hl_nums = hl_nums
        self.min_err = min_err
        self.max_err = max_err

    # @memoize
    # model.predict有小偏差，可能是exp的e和elu的e不一致
    def predict(self, x):
        for i in range(self.hl_nums):
            x = sigmoid(x * self.weights[i * 2] + self.weights[i * 2 + 1])
        return (x * self.weights[-2] + self.weights[-1])[0, 0]

    @staticmethod
    def init_by_dict(d: dict):
        weights_mat = [np.mat(weight) for weight in d['weights']]
        return AbstractNN(weights_mat, d['hl_nums'], d['min_err'], d['max_err'])


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Region):
            return obj.__dict__
        elif isinstance(obj, Geohash):
            return obj.save_to_dict()
        elif isinstance(obj, AbstractNN):
            return obj.__dict__
        elif isinstance(obj, SBRIN):
            return obj.save_to_dict()
        elif isinstance(obj, BlockRange):
            return obj.__dict__
        elif isinstance(obj, Meta):
            return obj.__dict__
        else:
            return super(MyEncoder, self).default(obj)


class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)

    def dict_to_object(self, d):
        if d.__contains__("weights") and d.__contains__("min_err") and d.__contains__("max_err"):
            t = AbstractNN.init_by_dict(d)
        elif len(d.keys()) == 4 and d.__contains__("bottom") and d.__contains__("up") \
                and d.__contains__("left") and d.__contains__("right"):
            t = Region.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "Geohash":
            t = Geohash.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "SBRIN":
            t = SBRIN.init_by_dict(d)
        elif d.__contains__("itemoffset"):
            t = BlockRange.init_by_dict(d)
        elif d.__contains__("pages_per_range"):
            t = Meta.init_by_dict(d)
        else:
            t = d
        return t


# @profile(precision=8)
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/trip_data_1_10w_sorted.npy'
    model_path = "model/sbrin_10w_1/"
    index = SBRIN(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        data_list = np.load(data_path).tolist()
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(data_list=data_list, threshold_number=1000, data_precision=6, region=Region(40, 42, -75, -73),
                    use_threshold=False,
                    threshold=20,
                    core=[1, 128, 1],
                    train_step=5000,
                    batch_num=16,
                    learning_rate=0.1,
                    retrain_time_limit=2,
                    thread_pool_size=10,
                    save_nn=True,
                    weight=1)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        index.save()
    # logging.info("Index size: %s" % index.size())
    # path = '../../data/trip_data_1_point_query.csv'
    # point_query_df = pd.read_csv(path, usecols=[1, 2, 3])[13:]
    # point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    # start_time = time.time()
    # results = index.point_query(point_query_list)
    # end_time = time.time()
    # search_time = (end_time - start_time) / len(point_query_list)
    # print("Point query time ", search_time)
    # np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    # path = '../../data/trip_data_1_range_query.csv'
    # range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    # range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    # start_time = time.time()
    # results = index.range_query(range_query_list)
    # end_time = time.time()
    # search_time = (end_time - start_time) / len(range_query_list)
    # print("Range query time ", search_time)
    # np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    # path = '../../data/trip_data_1_knn_query.csv'
    # knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    # knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    # start_time = time.time()
    # results = index.knn_query(knn_query_list)
    # end_time = time.time()
    # search_time = (end_time - start_time) / len(knn_query_list)
    # print("KNN query time ", search_time)
    # np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    insert_data_list = np.load("../../data/trip_data_1_10w.npy").tolist()
    start_time = time.time()
    index.update(insert_data_list)
    end_time = time.time()
    print("Update time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
