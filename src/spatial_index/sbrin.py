import copy
import gc
import logging
import math
import multiprocessing
import os
import sys
import time

import line_profiler
import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.learned_model_sbrin import TrainedNN
from src.spatial_index.common_utils import Region, binary_search_less_max, get_nearest_none, sigmoid, \
    biased_search_almost, biased_search, get_mbr_by_points
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex
from src.experiment.common_utils import load_data, Distribution

RA_PAGES = 256
PAGE_SIZE = 4096
HR_SIZE = 8 + 1 + 2 + 4 + 1  # 16
CR_SIZE = 8 * 4 + 2 + 1  # 35
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
ITEMS_PER_RA = RA_PAGES * int(PAGE_SIZE / ITEM_SIZE)


# TODO 检索的时候要检索crs
class SBRIN(SpatialIndex):
    def __init__(self, model_path=None, meta=None, history_ranges=None, current_ranges=None):
        super(SBRIN, self).__init__("SBRIN")
        self.index_entries = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # meta page由meta组成
        # version
        # last_hr: 新增：最后一个hr的指针
        # last_cr: 新增：最后一个hr的指针
        # threshold_number: 新增：hr的数据范围，也是hr分裂的索引项数量阈值
        # threshold_length: 新增：hr分裂的geohash长度阈值
        # threshold_err: 新增：hr重新训练model的误差阈值
        # threshold_summary: 新增：cr的数据范围，也是cr统计mbr的索引项数量阈值
        # threshold_merge: 新增：cr合并的cr数量阈值
        # geohash: 新增：对应L = geohash.sum_bits，索引项geohash编码实际长度
        self.meta = meta
        # history range pages由多个hr分页组成
        # value: 改动：L长度的整型geohash
        # length: 新增：geohash的实际length
        # number: 新增：range范围内索引项的数据量
        # model: 新增：learned indices
        # state: 新增：状态，1=inefficient
        # scope: 优化计算所需
        # value_diff: 优化计算所需：下一个hr value - hr value
        self.history_ranges = history_ranges
        # current range pages由多个cr分页组成
        # value: 改动：mbr
        # number: 新增：range范围内索引项的数据量
        # state: 新增：状态，1=full, 2=outdated
        self.current_ranges = current_ranges
        # 训练所需：
        self.is_gpu = None
        self.weight = None
        self.cores = None
        self.train_step = None
        self.batch_num = None
        self.learning_rate = None
        self.retrain_state = 0

    def build(self, data_list, is_sorted, threshold_number, data_precision, region, threshold_err,
              threshold_summary, threshold_merge,
              is_new, is_gpu, weight, core, train_step, batch_num, learning_rate, use_threshold, threshold,
              retrain_time_limit, thread_pool_size):
        """
        构建SBRIN
        1. order data by geohash
        2. build SBRIN
        2.1. init hr
        2.2. quartile recursively
        2.3. create sbrin
        2.4. reorganize index entries
        3. build learned model
        """
        self.is_gpu = is_gpu
        self.weight = weight
        self.cores = core
        self.train_step = train_step
        self.batch_num = batch_num
        self.learning_rate = learning_rate
        # 1. order data by geohash
        geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        if is_sorted:
            data_list = data_list.tolist()
        else:
            data_list = [(data_list[i][0], data_list[i][1], geohash.encode(data_list[i][0], data_list[i][1]), i)
                         for i in range(len(data_list))]
            data_list = sorted(data_list, key=lambda x: x[2])
        # 2. build SBRIN
        # 2.1. init hr
        n = len(data_list)
        tmp_stack = [(0, 0, n, 0, region)]
        result_list = []
        threshold_length = region.get_max_depth_by_region_and_precision(precision=data_precision) * 2
        # 2.2. quartile recursively
        while len(tmp_stack):
            cur = tmp_stack.pop(-1)
            if cur[2] > threshold_number and cur[1] < threshold_length:
                child_regions = cur[4].split()
                l_key = cur[3]
                r_key = cur[3] + cur[2] - 1
                tmp_l_key = l_key
                child_list = [None] * 4
                length = cur[1] + 2
                r_bound = cur[0]
                for i in range(4):
                    value = r_bound
                    r_bound = cur[0] + (i + 1 << geohash.sum_bits - length)
                    tmp_r_key = binary_search_less_max(data_list, 2, r_bound, tmp_l_key, r_key)
                    child_list[i] = (value, length, tmp_r_key - tmp_l_key + 1, tmp_l_key, child_regions[i])
                    tmp_l_key = tmp_r_key + 1
                tmp_stack.extend(child_list[::-1])  # 倒着放入init中，保持顺序
            else:
                # 把不需要分裂的hr加入结果list，加入的时候顺序为[左上，右下，左上，右上]的逆序，因为堆栈
                result_list.append(cur)
        # 2.3. create sbrin
        result_len = len(result_list)
        self.meta = Meta(result_len - 1, -1, threshold_number, threshold_length, threshold_err, threshold_summary,
                         threshold_merge, geohash)
        region_offset = pow(10, -data_precision - 1)
        self.history_ranges = [HistoryRange(result_list[i][0], result_list[i][1], result_list[i][2], None, 0,
                                            result_list[i][4].up_right_less_region(region_offset),
                                            2 << geohash.sum_bits - result_list[i][1] - 1) for i in range(result_len)]
        self.current_ranges = []
        self.create_cr()
        # 2.4. reorganize index entries
        result_data_list = []
        for i in range(result_len):
            result_data_list.extend(data_list[result_list[i][3]: result_list[i][3] + result_list[i][2]])
            result_data_list.extend([(0, 0, 0, 0)] * (threshold_number - result_list[i][2]))
        self.index_entries = result_data_list
        # 3. build learned model
        self.build_nn_multiprocess(is_new, is_gpu, weight, core, train_step, batch_num, learning_rate,
                                   use_threshold, threshold, retrain_time_limit, thread_pool_size)

    def build_nn_multiprocess(self, is_new, is_gpu, weight, core, train_step, batch_num, learning_rate,
                              use_threshold, threshold, retrain_time_limit, thread_pool_size):
        multiprocessing.set_start_method('spawn', force=True)  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()
        for i in range(self.meta.last_hr + 1):
            hr = self.history_ranges[i]
            # 训练数据为左下角点+分区数据+右上角点
            inputs = [j[2] for j in
                      self.index_entries[i * self.meta.threshold_number:i * self.meta.threshold_number + hr.number]]
            inputs.insert(0, hr.value)
            inputs.append(hr.value + hr.value_diff)
            data_num = hr.number + 2
            labels = list(range(data_num))
            batch_size = 2 ** math.ceil(math.log(data_num / batch_num, 2))
            if batch_size < 1:
                batch_size = 1
            # batch_size = batch_num
            pool.apply_async(build_nn, (self.model_path, i, inputs, labels, is_new, is_gpu,
                                        weight, core, train_step, batch_size, learning_rate,
                                        use_threshold, threshold, retrain_time_limit, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.history_ranges[key].model = value

    def reorganize_data_old(self):
        """
        把数据存在predict的地方，如果pre已有数据：
        1. pre处数据的geohash==数据本身的geohash，说明数据重复，则找到离pre最近的[pre-maxerr, pre-minerr]范围内的None来存储
        2. pre处数据的geohash!=数据本身的geohash，说明本该属于数据的位置被其他数据占用了，为了保持有序，找None的过程只往一边走
        存在问题：这种重构相当于在存储数据的时候依旧保持数据分布的稀疏性，但是密集的地方后续往往更加密集，导致这些地方的数据存储位置更加紧张
        这个问题往往在大数据量或误差大或分布不均匀的hr更容易出现，即最后"超出边界"的报错
        """
        hr_len = len(self.history_ranges)
        result_data_list = [None] * hr_len * self.meta.threshold_number
        for i in range(hr_len):
            hr = self.history_ranges[i]
            hr.model.output_min = self.meta.threshold_number * i
            hr.model.output_max = self.meta.threshold_number * (i + 1) - 1
            hr_first_key = i * self.meta.threshold_number
            for i in range(hr_first_key, hr_first_key + hr.number):
                pre = hr.model.predict(self.index_entries[i][2])
                if result_data_list[pre] is None:
                    result_data_list[pre] = self.index_entries[i]
                else:
                    # 重复数据处理：写入误差范围内离pre最近的None里
                    if result_data_list[pre][2] == self.index_entries[i][2]:
                        l_bound = max(pre - hr.model.max_err, hr.model.output_min)
                        r_bound = min(pre - hr.model.min_err, hr.model.output_max)
                    else:  # 非重复数据，但是整型部分重复，或被重复数据取代了位置
                        if result_data_list[pre][2] > self.index_entries[i][2]:
                            l_bound = max(pre - hr.model.max_err, hr.model.output_min)
                            r_bound = pre
                        else:
                            l_bound = pre
                            r_bound = min(pre - hr.model.min_err, hr.model.output_max)
                    key = get_nearest_none(result_data_list, pre, l_bound, r_bound)
                    if key is None:
                        # 超出边界是因为大量的数据相互占用导致误差放大
                        print("超出边界")
                    else:
                        result_data_list[key] = self.index_entries[i]
        self.index_entries = result_data_list

    def insert_single(self, point):
        # 1. encode p to geohash and create index entry(geohash, x, y, pointer)
        point.insert(-1, self.meta.geohash.encode(point[0], point[1]))
        # 2. insert into cr
        self.current_ranges[-1].number += 1
        self.index_entries.append(tuple(point))
        # 3. parallel transactions
        self.get_sum_up_full_cr()
        self.post_sum_up_full_cr()
        self.get_merge_outdated_cr()
        self.post_merge_outdated_cr()
        self.post_retrain_inefficient_model()

    def create_cr(self):
        self.current_ranges.append(CurrentRange(value=None, number=0, state=0))
        self.meta.last_cr += 1

    def get_sum_up_full_cr(self):
        """
        监听last cr的number，如果超过ts_summary，则设为full状态，并追加新的cr
        """
        if self.current_ranges[-1].number >= self.meta.threshold_summary:
            self.current_ranges[-1].state = 1
            self.create_cr()

    def post_sum_up_full_cr(self):
        """
        获取所有full的cr，统计MBR
        """
        # full_crs = [cr for cr in self.current_ranges if cr.state == 1]
        cr_key = self.meta.last_cr
        while cr_key >= 0:
            cr = self.current_ranges[cr_key]
            if cr.state == 1:
                first_key = (self.meta.last_hr + 1) * self.meta.threshold_number + cr_key * self.meta.threshold_summary
                self.current_ranges[cr_key].value = get_mbr_by_points(
                    self.index_entries[first_key:first_key + self.meta.threshold_summary])
                self.current_ranges[cr_key].state = 0
                break
            cr_key -= 1

    def get_merge_outdated_cr(self):
        """
        监听cr数量，如果超过ts_merge(outdated)，则把前ts_merge个cr设为outdated状态
        """
        if self.meta.last_cr >= self.meta.threshold_merge:
            for cr in self.current_ranges[:self.meta.threshold_merge]:
                cr.state = 2

    def post_merge_outdated_cr(self):
        """
        获取所有outdated的cr，合并其内前ts_merge个cr到hr，并删除这些cr对应的对象
        """
        # outdated_crs = [cr for cr in self.current_ranges if cr.state == 2]
        if self.current_ranges[0].state == 2:
            # 1. order index entries in outdated crs(first ts_merge cr)
            old_data_len = self.meta.threshold_merge * self.meta.threshold_summary
            first_key = (self.meta.last_hr + 1) * self.meta.threshold_number
            old_data = sorted(self.index_entries[first_key:first_key + old_data_len], key=lambda x: x[2])
            # 2. merge index entries into hrs
            hr_key = self.binary_search_less_max(old_data[0][2], 0, self.meta.last_hr) + 1
            tmp_l_key = 0
            tmp_r_key = 0
            while tmp_r_key < old_data_len:
                if hr_key <= self.meta.last_hr:
                    if self.history_ranges[hr_key].value > old_data[tmp_r_key][2]:
                        tmp_r_key += 1
                    else:
                        if tmp_r_key - tmp_l_key > 0:
                            self.update_hr(hr_key - 1, old_data[tmp_l_key:tmp_r_key])
                            tmp_l_key = tmp_r_key
                        hr_key += 1
                else:
                    self.update_hr(hr_key - 1, old_data[tmp_r_key:])
                    break
            # 3. delete crs/index entries
            del self.current_ranges[:self.meta.threshold_merge]
            self.meta.last_cr -= self.meta.threshold_merge
            del self.index_entries[first_key:first_key + old_data_len]

    def get_retrain_inefficient_model(self, hr):
        """
        在模型更新时，监听误差范围，如果超过ts_err(inefficient)，则设为inefficient状态
        """
        if hr.model.max_err - hr.model.min_err >= self.meta.threshold_err:
            hr.state = 1
            self.retrain_state = 1

    def post_retrain_inefficient_model(self):
        """
        重训练所有低效状态的HR
        """
        if self.retrain_state:
            for i in range(self.meta.last_hr + 1):
                hr = self.history_ranges[i]
                if hr.state:
                    inputs = [j[2] for j in self.index_entries[
                                            i * self.meta.threshold_number:i * self.meta.threshold_number + hr.number]]
                    inputs.insert(0, hr.value)
                    inputs.append(hr.value + hr.value_diff)
                    data_num = hr.number + 2
                    labels = list(range(data_num))
                    batch_size = 2 ** math.ceil(math.log(data_num / self.batch_num, 2))
                    if batch_size < 1:
                        batch_size = 1
                    tmp_index = TrainedNN(self.model_path, str(i), inputs, labels, True, self.is_gpu, self.weight,
                                          self.cores, self.train_step, batch_size, self.learning_rate,
                                          False, None, None)
                    tmp_index.train_simple(hr.model.matrices)
                    hr.model = AbstractNN(tmp_index.matrices, hr.model.hl_nums,
                                          math.ceil(tmp_index.min_err),
                                          math.ceil(tmp_index.max_err))
                    hr.state = 0
            self.retrain_state = 0

    def update_hr(self, hr_key, points):
        """
        update hr by points
        """
        hr = self.history_ranges[hr_key]
        # quicksort->merge_sorted_array->sorted => 50:2:1
        # merge_sorted_array(self.index_entries, 2, hr.key[0], hr.key[1], points)
        offset = hr_key * self.meta.threshold_number
        points.extend(self.index_entries[offset:offset + hr.number])
        points_len = len(points)
        points = sorted(points, key=lambda x: x[2])
        if points_len > self.meta.threshold_number and hr.length < self.meta.threshold_length:
            # split hr
            self.split_hr(hr, hr_key, offset, points)
        else:
            # update hr metadata
            hr.number = points_len
            hr.max_key = points_len - 1
            hr.model_update([point[2] for point in points])
            self.get_retrain_inefficient_model(hr)
            # update index entries
            self.index_entries[offset:offset + points_len] = points

    def split_hr(self, hr, hr_key, offset, points):
        # 1. create child hrs, of which model is inherited from parent hr and update err by inherited index entries
        length = hr.length + 2
        value_diff = hr.value_diff >> 2
        region_offset = pow(10, -self.meta.geohash.data_precision - 1)
        child_regs = hr.scope.up_right_more_region(region_offset).split()
        last_key = len(points) - 1
        tmp_l_key = 0
        child_hrs = [None] * 4
        r_bound = hr.value
        child_ies = []
        for i in range(3):
            value = r_bound
            r_bound = hr.value + (i + 1) * value_diff
            tmp_r_key = binary_search_less_max(points, 2, r_bound, tmp_l_key, last_key)
            number = tmp_r_key - tmp_l_key + 1
            child_ies.extend(points[tmp_l_key: tmp_r_key + 1])
            child_ies.extend([(0, 0, 0, 0)] * (self.meta.threshold_number - number))
            # TODO: 当前model直接继承，需要改为计算得到父model的1/4部分
            child_hr = HistoryRange(value, length, number, copy.copy(hr.model), 0,
                                    child_regs[i].up_right_less_region(region_offset), value_diff)
            child_hr.model_update([point[2] for point in points[tmp_l_key: tmp_r_key + 1]])
            self.get_retrain_inefficient_model(child_hr)
            child_hrs[3 - i] = child_hr
            tmp_l_key = tmp_r_key + 1
        number = last_key - tmp_l_key + 1
        child_ies.extend(points[tmp_l_key:])
        child_ies.extend([(0, 0, 0, 0)] * (self.meta.threshold_number - number))
        child_hr = HistoryRange(r_bound, length, number, hr.model, 0, child_regs[i].up_right_less_region(region_offset),
                                value_diff)
        child_hr.model_update([point[2] for point in points[tmp_l_key:]])
        self.get_retrain_inefficient_model(child_hr)
        child_hrs[0] = child_hr
        # 2. delete old hr
        del self.history_ranges[hr_key]
        # 3. insert child hrs
        for child_hr in child_hrs:
            self.history_ranges.insert(hr_key, child_hr)
        # 4. update meta
        self.meta.last_hr += 3
        # 5. update index entries
        self.index_entries = self.index_entries[:offset] + child_ies + \
                             self.index_entries[offset + self.meta.threshold_number:]

    def point_query_hr(self, point):
        """
        根据geohash找到所在的hr的key
        1. 计算geohash对应到hr的geohash_int
        2. 找到比geohash_int小的最大值即为geohash所在的hr
        """
        return self.binary_search_less_max(point, 0, self.meta.last_hr)

    def range_query_hr(self, point1, point2):
        """
        根据geohash1/geohash2找到之间所有hr的key以及和window的位置关系
        1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
        3. 根据tgt_geohash分组并合并position
        """
        # 1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        hr_key1 = self.binary_search_less_max(point1, 0, self.meta.last_hr)
        hr_key2 = self.binary_search_less_max(point2, hr_key1, self.meta.last_hr)
        if hr_key1 == hr_key2:
            return {hr_key1: 15}
        else:
            max_length = max(self.history_ranges[hr_key1].length, self.history_ranges[hr_key2].length)
            org_geohash_list = self.meta.geohash.ranges_by_int(point1, point2, max_length)
            # 2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
            # 3. 根据tgt_geohash分组并合并position
            size = len(org_geohash_list) - 1
            i = 1
            tgt_geohash_dict = {hr_key1: org_geohash_list[0][1],
                                hr_key2: org_geohash_list[-1][1]}
            while True:
                if self.history_ranges[hr_key1].value > org_geohash_list[i][0]:
                    key = hr_key1 - 1
                    pos = org_geohash_list[i][1]
                    if self.history_ranges[key].length > max_length:
                        tgt_geohash_dict[key] = pos
                        tgt_geohash_dict[key + 1] = pos
                        tgt_geohash_dict[key + 2] = pos
                        tgt_geohash_dict[key + 3] = pos
                    else:
                        tgt_geohash_dict[key] = tgt_geohash_dict.get(key, 0) | org_geohash_list[i][1]
                    i += 1
                    if i >= size:
                        break
                else:
                    hr_key1 += 1
                    if hr_key1 > self.meta.last_hr:
                        tgt_geohash_dict[hr_key2] = tgt_geohash_dict[hr_key2] | org_geohash_list[i][1]
                        break
            return tgt_geohash_dict
            # 前缀匹配太慢：时间复杂度=O(len(window对应的geohash个数)*(j-i))

    def knn_query_hr(self, center_hr_key, point1, point2, point3):
        """
        根据geohash1/geohash2找到之间所有hr的key以及和window的位置关系，并基于和point3距离排序
        1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
        3. 根据tgt_geohash分组并合并position
        4. 计算每个tgt_geohash和point3的距离，并进行降序排序
        """
        # 1. 通过geohash_int1/geohash_int2找到window对应的所有org_geohash和对应window的position
        hr_key1 = self.biased_search_less_max(point1, center_hr_key, 0, center_hr_key)
        hr_key2 = self.biased_search_less_max(point2, center_hr_key, center_hr_key, self.meta.last_hr)
        if hr_key1 == hr_key2:
            return [[hr_key1, 15, 0]]
        else:
            max_length = max(self.history_ranges[hr_key1].length, self.history_ranges[hr_key2].length)
            org_geohash_list = self.meta.geohash.ranges_by_int(point1, point2, max_length)
            # 2. 通过前缀匹配过滤org_geohash来找到tgt_geohash
            # 3. 根据tgt_geohash分组并合并position
            size = len(org_geohash_list) - 1
            i = 1
            tgt_geohash_dict = {hr_key1: org_geohash_list[0][1],
                                hr_key2: org_geohash_list[-1][1]}
            while True:
                if self.history_ranges[hr_key1].value > org_geohash_list[i][0]:
                    key = hr_key1 - 1
                    pos = org_geohash_list[i][1]
                    if self.history_ranges[key].length > max_length:
                        tgt_geohash_dict[key] = pos
                        tgt_geohash_dict[key + 1] = pos
                        tgt_geohash_dict[key + 2] = pos
                        tgt_geohash_dict[key + 3] = pos
                    else:
                        tgt_geohash_dict[key] = tgt_geohash_dict.get(key, 0) | org_geohash_list[i][1]
                    i += 1
                    if i >= size:
                        break
                else:
                    hr_key1 += 1
                    if hr_key1 > self.meta.last_hr:
                        tgt_geohash_dict[hr_key2] = tgt_geohash_dict[hr_key2] | org_geohash_list[i][1]
                        break
            # 4. 计算每个tgt_geohash和point3的距离，并进行降序排序
            return sorted([[tgt_geohash,
                            tgt_geohash_dict[tgt_geohash],
                            self.history_ranges[tgt_geohash].scope.get_min_distance_pow_by_point_list(point3)]
                           for tgt_geohash in tgt_geohash_dict], key=lambda x: x[2])

    def binary_search_less_max(self, x, left, right):
        """
        二分查找比x小的最大值
        优化: 循环->二分->最左匹配:15->1->0.75
        """
        while left <= right:
            mid = (left + right) // 2
            if self.history_ranges[mid].value <= x:
                # 最左匹配
                if self.meta.last_hr == mid or self.history_ranges[mid + 1].value > x:
                    return mid
                left = mid + 1
            else:
                right = mid - 1

    def biased_search_less_max(self, x, mid, left, right):
        """
        二分查找比x小的最大值，指定初始mid
        优化: 二分->biased二分:3->1
        """
        while left <= right:
            if self.history_ranges[mid].value <= x:
                if self.meta.last_hr == mid or self.history_ranges[mid + 1].value > x:
                    return mid
                left = mid + 1
            else:
                right = mid - 1
            mid = (left + right) // 2

    def point_query_single(self, point):
        """
        1. compute geohash from x/y of points
        2. find hr within geohash by sbrin.point_query
        3. predict by leaf model
        4. biased search in scope [pre - max_err, pre + min_err]
        """
        # 1. compute geohash from x/y of point
        gh = self.meta.geohash.encode(point[0], point[1])
        # 2. find hr within geohash by sbrin.point_query
        hr_key = self.point_query_hr(gh)
        hr = self.history_ranges[hr_key]
        if hr.number == 0:
            return None
        else:
            # 3. predict by leaf model
            pre = hr.model_predict(gh)
            offset = hr_key * self.meta.threshold_number
            # 4. biased search in scope [pre - max_err, pre + min_err]
            return [self.index_entries[key][3]
                    for key in biased_search(self.index_entries, 2, gh, offset + pre,
                                             offset + max(pre - hr.model.max_err, 0),
                                             offset + min(pre - hr.model.min_err, hr.max_key))]

    def range_query_single(self, window):
        """
        1. compute geohash from window_left and window_right
        2. get all relative hrs with key and relationship
        3. get min_geohash and max_geohash of every hr for different relation
        4. predict min_key/max_key by nn
        5. filter all the point of scope[min_key/max_key] by range.contain(point)
        主要耗时间：range_query_hr/nn predict/精确过滤: 15/24/37.6
        """
        # 1. compute geohash of window_left and window_right
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        # 2. get all relative hrs with key and relationship
        hr_list = self.range_query_hr(gh1, gh2)
        result = []
        # 3. get min_geohash and max_geohash of every hr for different relation
        for hr_key in hr_list:
            hr = self.history_ranges[hr_key]
            if hr.number == 0:  # hr is empty
                continue
            position = hr_list[hr_key]
            if position == 0:  # window contain hr
                key_left = hr_key * self.meta.threshold_number
                result.extend(list(range(key_left, key_left + hr.number)))
            else:
                # wrong child hr from range_by_int
                is_valid = valid_position_funcs[position](hr.scope, window)
                if not is_valid:
                    continue
                # if-elif-else->lambda, 30->4
                gh_new1, gh_new2, compare_func = range_position_funcs[position](hr.scope, window, gh1, gh2,
                                                                                self.meta.geohash)
                # 4 predict min_key/max_key by nn
                offset = hr_key * self.meta.threshold_number
                if gh_new1:
                    pre1 = hr.model_predict(gh_new1)
                    l_bound1 = max(pre1 - hr.model.max_err, 0)
                    r_bound1 = min(pre1 - hr.model.min_err, hr.max_key)
                    key_left = min(biased_search_almost(self.index_entries, 2, gh_new1,
                                                        pre1 + offset, l_bound1 + offset, r_bound1 + offset))
                else:
                    key_left = offset
                if gh_new2:
                    pre2 = hr.model_predict(gh_new2)
                    l_bound2 = max(pre2 - hr.model.max_err, 0)
                    r_bound2 = min(pre2 - hr.model.min_err, hr.number)
                    key_right = max(biased_search_almost(self.index_entries, 2, gh_new2,
                                                         pre2 + offset, l_bound2 + offset, r_bound2 + offset))
                else:
                    key_right = key_left + hr.number
                # 5 filter all the point of scope[min_key/max_key] by range.contain(point)
                # 优化: region.contain->compare_func不同位置的点做不同的判断: 638->474mil
                result.extend([ie[3] for ie in self.index_entries[key_left:key_right] if compare_func(ie)])
        return result

    def knn_query_single(self, knn):
        """
        1. get the nearest key of query point
        2. get the nn points to create range query window
        3. filter point by distance
        主要耗时间：knn_query_hr/nn predict/精确过滤: 6.1/30/40.5
        """
        x, y, k = knn
        # 1. get the nearest key of query point
        qp_g = self.meta.geohash.encode(x, y)
        qp_hr_key = self.point_query_hr(qp_g)
        qp_hr = self.history_ranges[qp_hr_key]
        # if hr is empty, TODO
        if qp_hr.number == 0:
            return []
        # if model, qp_ie_key = point_query(geohash)
        else:
            offset = qp_hr_key * self.meta.threshold_number
            pre = qp_hr.model_predict(qp_g)
            l_bound = max(pre - qp_hr.model.max_err, 0)
            r_bound = min(pre - qp_hr.model.min_err, qp_hr.max_key)
            qp_ie_key = biased_search_almost(self.index_entries, 2, qp_g,
                                             pre + offset, l_bound + offset, r_bound + offset)[0]
        # 2. get the n points to create range query window
        # TODO: 两种策略，一种是左右找一半，但是如果跳跃了，window很大；
        #  还有一种是两边找n，减少跳跃，使window变小，当前是第二种
        tp_ie_list = [self.index_entries[qp_ie_key]]
        cur_ie_key = qp_ie_key + 1
        cur_hr = qp_hr
        cur_hr_key = qp_hr_key
        i = k
        while i > 0:
            right_ie_len = cur_hr.number - cur_ie_key % self.meta.threshold_number
            if right_ie_len >= i:
                tp_ie_list.extend(self.index_entries[cur_ie_key:cur_ie_key + i])
                break
            else:
                tp_ie_list.extend(self.index_entries[cur_ie_key:cur_ie_key + right_ie_len])
                i -= right_ie_len
                cur_hr_key += 1
                if cur_hr_key > self.meta.last_hr:
                    break
                cur_hr = self.history_ranges[cur_hr_key]
                cur_ie_key = cur_hr_key * self.meta.threshold_number
        cur_ie_key = qp_ie_key
        cur_hr_key = qp_hr_key
        i = k
        while i > 0:
            left_ie_len = cur_ie_key % self.meta.threshold_number
            if left_ie_len >= i:
                tp_ie_list.extend(self.index_entries[cur_ie_key - i:cur_ie_key])
                break
            else:
                tp_ie_list.extend(self.index_entries[cur_ie_key - left_ie_len:cur_ie_key])
                i -= left_ie_len
                cur_hr_key -= 1
                if cur_hr_key < 0:
                    break
                cur_ie_key = cur_hr_key * self.meta.threshold_number + self.history_ranges[cur_hr_key].number
        tp_list = sorted([[(tp_ie[0] - x) ** 2 + (tp_ie[1] - y) ** 2, tp_ie[3]] for tp_ie in tp_ie_list])[:k]
        max_dist = tp_list[-1][0]
        if max_dist == 0:
            return [tp[1] for tp in tp_list]
        max_dist_pow = max_dist ** 0.5
        window = [y - max_dist_pow, y + max_dist_pow, x - max_dist_pow, x + max_dist_pow]
        # 处理超出边界的情况
        self.meta.geohash.region.clip_region(window, self.meta.geohash.data_precision)
        gh1 = self.meta.geohash.encode(window[2], window[0])
        gh2 = self.meta.geohash.encode(window[3], window[1])
        tp_window_hrs = self.knn_query_hr(qp_hr_key, gh1, gh2, knn)
        tp_list = []
        for tp_window_hr in tp_window_hrs:
            if tp_window_hr[2] > max_dist:
                break
            hr = self.history_ranges[tp_window_hr[0]]
            if hr.number == 0:  # hr is empty
                continue
            position = tp_window_hr[1]
            if position == 0:  # window contain hr
                key_left = tp_window_hr[0] * self.meta.threshold_number
                tmp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                            for ie in self.index_entries[key_left:key_left + hr.number]]
            else:
                # wrong child hr from range_by_int
                is_valid = valid_position_funcs[position](hr.scope, window)
                if not is_valid:
                    continue
                gh_new1, gh_new2, compare_func = range_position_funcs[tp_window_hr[1]](hr.scope, window, gh1, gh2,
                                                                                       self.meta.geohash)
                offset = tp_window_hr[0] * self.meta.threshold_number
                if gh_new1:
                    pre1 = hr.model_predict(gh_new1)
                    l_bound1 = max(pre1 - hr.model.max_err, 0)
                    r_bound1 = min(pre1 - hr.model.min_err, hr.max_key)
                    key_left = min(biased_search_almost(self.index_entries, 2, gh_new1,
                                                        pre1 + offset, l_bound1 + offset, r_bound1 + offset))
                else:
                    key_left = offset
                if gh_new2:
                    pre2 = hr.model_predict(gh_new2)
                    l_bound2 = max(pre2 - hr.model.max_err, 0)
                    r_bound2 = min(pre2 - hr.model.min_err, hr.number)
                    key_right = max(biased_search_almost(self.index_entries, 2, gh_new2,
                                                         pre2 + offset, l_bound2 + offset, r_bound2 + offset))
                else:
                    key_right = offset + hr.number
                # 3. filter point by distance
                tmp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                            for ie in self.index_entries[key_left:key_right] if compare_func(ie)]
            if len(tmp_list) > 0:
                tp_list.extend(tmp_list)
                tp_list = sorted(tp_list)[:k]
                max_dist = tp_list[-1][0]
        return [tp[1] for tp in tp_list]

    def save(self):
        assert self.meta.threshold_number < 2 ** 16, "threshold_number exceed the store size int16"
        sbrin_meta = np.array((self.meta.last_hr, self.meta.last_cr,
                               self.meta.threshold_number, self.meta.threshold_length,
                               self.meta.threshold_err, self.meta.threshold_summary, self.meta.threshold_merge,
                               self.meta.geohash.data_precision,
                               self.meta.geohash.region.bottom, self.meta.geohash.region.up,
                               self.meta.geohash.region.left, self.meta.geohash.region.right,
                               self.is_gpu, self.weight, self.train_step, self.batch_num, self.learning_rate),
                              dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i2'), ("3", 'i2'), ("4", 'i2'), ("5", 'i2'),
                                     ("6", 'i2'), ("7", 'i1'),
                                     ("8", 'f8'), ("9", 'f8'), ("10", 'f8'), ("11", 'f8'),
                                     ("12", 'i1'), ("13", 'f4'), ("14", 'i2'), ("15", 'i2'), ("16", 'f4')])
        sbrin_models = np.array([hr.model for hr in self.history_ranges])
        sbrin_hrs = np.array([(hr.value, hr.length, hr.number, hr.state, hr.value_diff,
                               hr.scope.bottom, hr.scope.up, hr.scope.left, hr.scope.right)
                              for hr in self.history_ranges],
                             dtype=[("0", 'i8'), ("1", 'i1'), ("2", 'i2'), ("3", 'i1'), ("4", 'i8'),
                                    ("5", 'f8'), ("6", 'f8'), ("7", 'f8'), ("8", 'f8')])
        sbrin_crs = []
        for cr in self.current_ranges:
            if cr.value is None:
                cr_list = [-1, -1, -1, -1, cr.number, cr.state]
            else:
                cr_list = [cr.value.bottom, cr.value.up, cr.value.left, cr.value.right, cr.number, cr.state]
            sbrin_crs.append(tuple(cr_list))
        sbrin_crs = np.array(sbrin_crs, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'f8'), ("3", 'f8'),
                                               ("4", 'i2'), ("5", 'i1')])
        np.save(os.path.join(self.model_path, 'sbrin_meta.npy'), sbrin_meta)
        np.save(os.path.join(self.model_path, 'sbrin_model_cores.npy'), self.cores)
        np.save(os.path.join(self.model_path, 'sbrin_hrs.npy'), sbrin_hrs)
        np.save(os.path.join(self.model_path, 'sbrin_models.npy'), sbrin_models)
        np.save(os.path.join(self.model_path, 'sbrin_crs.npy'), sbrin_crs)
        index_entries = np.array(self.index_entries, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4')])
        np.save(os.path.join(self.model_path, 'sbrin_data.npy'), index_entries)

    def load(self):
        sbrin_meta = np.load(os.path.join(self.model_path, 'sbrin_meta.npy'), allow_pickle=True).item()
        sbrin_hrs = np.load(os.path.join(self.model_path, 'sbrin_hrs.npy'), allow_pickle=True)
        sbrin_models = np.load(os.path.join(self.model_path, 'sbrin_models.npy'), allow_pickle=True)
        sbrin_crs = np.load(os.path.join(self.model_path, 'sbrin_crs.npy'), allow_pickle=True)
        index_entries = np.load(os.path.join(self.model_path, 'sbrin_data.npy'), allow_pickle=True)
        region = Region(sbrin_meta[8], sbrin_meta[9], sbrin_meta[10], sbrin_meta[11])
        geohash = Geohash.init_by_precision(data_precision=sbrin_meta[7], region=region)
        self.meta = Meta(sbrin_meta[0], sbrin_meta[1], sbrin_meta[2], sbrin_meta[3], sbrin_meta[4], sbrin_meta[5],
                         sbrin_meta[6], geohash)
        self.cores = np.load(os.path.join(self.model_path, 'sbrin_model_cores.npy'), allow_pickle=True).tolist()
        self.is_gpu = bool(sbrin_meta[12])
        self.weight = sbrin_meta[13]
        self.train_step = sbrin_meta[14]
        self.batch_num = sbrin_meta[15]
        self.learning_rate = sbrin_meta[16]
        self.history_ranges = [
            HistoryRange(sbrin_hrs[i][0], sbrin_hrs[i][1], sbrin_hrs[i][2], sbrin_models[i], sbrin_hrs[i][3],
                         Region(sbrin_hrs[i][5], sbrin_hrs[i][6], sbrin_hrs[i][7], sbrin_hrs[i][8]),
                         sbrin_hrs[i][4]) for i in range(len(sbrin_hrs))]
        crs = []
        for i in range(len(sbrin_crs)):
            cr = sbrin_crs[i]
            if cr[0] == -1:
                region = None
            else:
                region = Region(cr[0], cr[1], cr[2], cr[3])
            crs.append(CurrentRange(region, cr[4], cr[5]))
        self.current_ranges = crs
        self.index_entries = index_entries.tolist()

    def size(self):
        """
        structure_size = sbrin_meta.npy + sbrin_hrs.npy + sbrin_models.npy + sbrin_crs.npy
        ie_size = sbrin_data.npy
        """
        # 实际上：
        # meta=os.path.getsize(os.path.join(self.model_path, "sbrin_meta.npy"))-128-64*3=1*2+2*7+4*4+8*4=64
        # hr=os.path.getsize(os.path.join(self.model_path, "sbrin_hrs.npy"))-128-64=hr_len*(1*2+2*1+8*6)=hr_len*52
        # model一致=os.path.getsize(os.path.join(self.model_path, "sbrin_models.npy"))-128=hr_len*model_size
        # cr=os.path.getsize(os.path.join(self.model_path, "sbrin_crs.npy"))-128-64=cr_len*(1*1+2*1+8*4)=cr_len*35
        # index_entries=os.path.getsize(os.path.join(self.model_path, "sbrin_data.npy"))-128
        # =hr_len*meta.threshold_number*(8*3+4)
        # 理论上：
        # meta只存last_hr/last_cr/5*ts/L=4+4+5*2+1=19
        # hr只存value/length/number/*model/state=hr_len*(8+1+2+4+1)=hr_len*16
        # cr只存value/number/state=cr_len*(8*4+2+1)=cr_len*35
        # index_entries为data_len*(8*3+4)=data_len*28
        data_len = sum([hr.number for hr in self.history_ranges]) + sum([cr.number for cr in self.current_ranges])
        hr_len = self.meta.last_hr + 1
        cr_len = self.meta.last_cr + 1
        return 19 + \
               hr_len * 16 + \
               os.path.getsize(os.path.join(self.model_path, "sbrin_models.npy")) - 128 + \
               cr_len * 35, data_len * 28

    def io(self):
        """
        假设查询条件和数据分布一致，io=获取meta的io+获取hr的io+获取cr的io+对应model的io+获取model内数据的io
        一次read_ahead可以加载meta+hr+cr和部分的model, 其他model需要第二次read_ahead，model内数据单独一次read_ahead
        先计算单个model的model io和data io，然后乘以model的数据量，最后除以总数据量，来计算整体的平均io
        """
        hr_len = self.meta.last_hr + 1
        meta_page_len = 1
        hr_page_len = math.ceil((self.meta.last_hr + 1) * HR_SIZE / PAGE_SIZE)
        cr_page_len = math.ceil((self.meta.last_cr + 1) * CR_SIZE / PAGE_SIZE)
        model_page_len = math.ceil((self.meta.last_hr + 1) * MODEL_SIZE / PAGE_SIZE)
        origin_page_len = meta_page_len + hr_page_len + cr_page_len + model_page_len
        # io when load model
        if origin_page_len < RA_PAGES:
            model_io_list = [1] * hr_len
        else:
            model_io_list = [1] * origin_page_len
            model_io_list.extend([2] * (hr_len - origin_page_len))
        # io when load data
        data_io_list = [math.ceil((hr.model.max_err - hr.model.min_err) / ITEMS_PER_RA) for hr in self.history_ranges]
        # compute avg io
        data_num_list = [hr.number for hr in self.history_ranges]
        io_list = [(model_io_list[i] + data_io_list[i]) * data_num_list[i] for i in range(hr_len)]
        return sum(io_list) / sum(data_num_list)

    def model_clear(self):
        """
        清除非最小误差的model
        """
        for i in range(self.meta.last_hr + 1):
            tmp_index = TrainedNN(self.model_path, str(i), None, None, None, None, None,
                                  None, None, None, None, None, None, None)
            tmp_index.clean_not_best_model_file()


# for query
valid_position_funcs = [
    lambda reg, window: None,
    lambda reg, window:  # right
    window[3] >= reg.left,
    lambda reg, window:  # left
    window[2] <= reg.right,
    lambda reg, window:  # left-right
    window[2] <= reg.right and reg.left <= window[3],
    lambda reg, window:  # up
    window[1] >= reg.bottom,
    lambda reg, window:  # up-right
    window[3] >= reg.left and window[1] >= reg.bottom,
    lambda reg, window:  # up-left
    window[2] <= reg.right and window[1] >= reg.bottom,
    lambda reg, window:  # up-left-right
    window[2] <= reg.right and reg.left <= window[3] and window[1] >= reg.bottom,
    lambda reg, window:  # bottom
    window[0] <= reg.up,
    lambda reg, window:  # bottom-right
    window[3] >= reg.left and window[0] <= reg.up,
    lambda reg, window:  # bottom-left
    window[2] <= reg.right and window[0] <= reg.up,
    lambda reg, window:  # bottom-left-right
    window[2] <= reg.right and reg.left <= window[3] and window[0] <= reg.up,
    lambda reg, window:  # bottom-up
    window[0] <= reg.up and reg.bottom <= window[1],
    lambda reg, window:  # bottom-up-right
    window[3] >= reg.left and reg.right and window[0] <= reg.up and reg.bottom <= window[1],
    lambda reg, window:  # bottom-up-left
    window[2] <= reg.right and window[0] <= reg.up and reg.bottom <= window[1],
    lambda reg, window:  # bottom-up-left-right
    window[2] <= reg.right and reg.left <= window[3] and window[0] <= reg.up and reg.bottom <= window[1]]
range_position_funcs = [
    lambda reg, window, gh1, gh2, geohash: (None, None, None),
    lambda reg, window, gh1, gh2, geohash: (  # right
        None,
        geohash.encode(window[3], reg.up),
        lambda x: window[3] >= x[0]),
    lambda reg, window, gh1, gh2, geohash: (  # left
        geohash.encode(window[2], reg.bottom),
        None,
        lambda x: window[2] <= x[0]),
    lambda reg, window, gh1, gh2, geohash: (  # left-right
        geohash.encode(window[2], reg.bottom),
        geohash.encode(window[3], reg.up),
        lambda x: window[2] <= x[0] <= window[3]),
    lambda reg, window, gh1, gh2, geohash: (  # up
        None,
        geohash.encode(reg.right, window[1]),
        lambda x: window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # up-right
        None,
        gh2,
        lambda x: window[3] >= x[0] and window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # up-left
        geohash.encode(window[2], reg.bottom),
        geohash.encode(reg.right, window[1]),
        lambda x: window[2] <= x[0] and window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # up-left-right
        geohash.encode(window[2], reg.bottom),
        gh2,
        lambda x: window[2] <= x[0] <= window[3] and window[1] >= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom
        geohash.encode(reg.left, window[0]),
        None,
        lambda x: window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-right
        geohash.encode(reg.left, window[0]),
        geohash.encode(window[3], reg.up),
        lambda x: window[3] >= x[0] and window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-left
        gh1,
        None,
        lambda x: window[2] <= x[0] and window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-left-right
        gh1,
        geohash.encode(window[3], reg.up),
        lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up
        geohash.encode(reg.left, window[0]),
        geohash.encode(reg.right, window[1]),
        lambda x: window[0] <= x[1] <= window[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up-right
        geohash.encode(reg.left, window[0]),
        gh2,
        lambda x: window[3] >= x[0] and window[0] <= x[1] <= window[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up-left
        gh1,
        geohash.encode(reg.right, window[1]),
        lambda x: window[2] <= x[0] and window[0] <= x[1] <= window[1]),
    lambda reg, window, gh1, gh2, geohash: (  # bottom-up-left-right
        gh1,
        gh2,
        lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1] <= window[1])]


# for train
def build_nn(model_path, model_key, inputs, labels, is_new, is_gpu, weight, core, train_step, batch_size,
             learning_rate, use_threshold, threshold, retrain_time_limit, tmp_dict=None):
    tmp_index = TrainedNN(model_path, str(model_key), inputs, labels, is_new, is_gpu, weight, core,
                          train_step, batch_size, learning_rate, use_threshold, threshold, retrain_time_limit)
    tmp_index.train()
    abstract_index = AbstractNN(tmp_index.matrices, len(core) - 1,
                                math.ceil(tmp_index.min_err),
                                math.ceil(tmp_index.max_err))
    del tmp_index
    gc.collect()
    tmp_dict[model_key] = abstract_index


class Meta:
    def __init__(self, last_hr, last_cr, threshold_number, threshold_length, threshold_err, threshold_summary,
                 threshold_merge, geohash):
        # BRIN
        # SBRIN
        self.last_hr = last_hr
        self.last_cr = last_cr
        self.threshold_number = threshold_number
        self.threshold_length = threshold_length
        self.threshold_err = threshold_err
        self.threshold_summary = threshold_summary
        self.threshold_merge = threshold_merge
        # self.L = L  # geohash.sum_bits
        # For compute
        self.geohash = geohash


class HistoryRange:
    def __init__(self, value, length, number, model, state, scope, value_diff):
        # BRIN
        self.value = value
        # SBRIN
        self.length = length
        self.number = number
        self.model = model
        self.state = state
        # For compute
        self.scope = scope
        self.value_diff = value_diff
        self.max_key = number - 1

    def model_predict(self, x):
        x = self.model.predict((x - self.value) / self.value_diff - 0.5)
        if x <= 0:
            return 0
        elif x >= 1:
            return self.max_key
        return int(self.max_key * x)

    def model_update(self, xs):
        if self.number:
            pres = self.model.predicts((np.mat(xs).T - self.value) / self.value_diff - 0.5)
            pres[pres < 0] = 0
            pres[pres > 1] = 1
            errs = pres * self.max_key - np.arange(self.number)
            self.model.min_err = math.ceil(errs.min())
            self.model.max_err = math.ceil(errs.max())
        else:
            self.model.min_err = 0
            self.model.max_err = 0


class CurrentRange:
    def __init__(self, value, number, state):
        # BRIN
        self.value = value
        # SBRIN
        self.number = number
        self.state = state
        # For compute


class AbstractNN:
    def __init__(self, matrices, hl_nums, min_err, max_err):
        self.matrices = matrices
        self.hl_nums = hl_nums
        self.min_err = min_err
        self.max_err = max_err

    # model.predict有小偏差，可能是exp的e和elu的e不一致
    def predict(self, x):
        for i in range(self.hl_nums):
            x = sigmoid(x * self.matrices[i * 2] + self.matrices[i * 2 + 1])
        return (x * self.matrices[-2] + self.matrices[-1])[0, 0]

    def predicts(self, xs):
        for i in range(self.hl_nums):
            xs = sigmoid(xs * self.matrices[i * 2] + self.matrices[i * 2 + 1])
        return (xs * self.matrices[-2] + self.matrices[-1]).T.A


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/sbrin_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    model_hdf_dir = os.path.join(model_path, "hdf/")
    if os.path.exists(model_hdf_dir) is False:
        os.makedirs(model_hdf_dir)
    index = SBRIN(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(Distribution.NYCT_10W_SORTED, 0)
        # 按照pagesize=4096, read_ahead=256, size(pointer)=4, size(x/y/g)=8, sbrin整体连续存, meta一个page, br分页存，model(2009大小)单独存
        # hr体积=value/length/number=16，一个page存256个hr
        # cr体积=value/number=35，一个page存117个cr
        # model体积=2009，一个page存2个model
        # data体积=x/y/g/key=8*3+4=28，一个page存146个data
        # 10w数据，[1000]参数下：大约有289个cr
        # 1meta page，289/256=2hr page，1cr page, 289/2=145model page，10w/146=685data page
        # 单次扫描IO=读取sbrin+读取对应model+读取model对应索引项=1+1+误差范围/146/256
        # 索引体积=meta+hrs+crs+model+索引项
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    threshold_number=1000,
                    data_precision=6,
                    region=Region(40, 42, -75, -73),
                    threshold_err=200,
                    threshold_summary=1000,
                    threshold_merge=5,
                    is_new=False,
                    is_gpu=True,
                    weight=1,
                    core=[1, 128],
                    train_step=5000,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=1,
                    thread_pool_size=6)
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
    path = '../../data/query/knn_query_nyct.npy'
    knn_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/table/trip_data_2_filter_10w.npy'
    insert_data_list = np.load(path, allow_pickle=True)[:, [10, 11, -1]]
    profile = line_profiler.LineProfiler(index.update_hr)
    profile.enable()
    index.insert(update_data_list.tolist())
    profile.disable()
    profile.print_stats()
    start_time = time.time()
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
