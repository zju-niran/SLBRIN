import logging
import math
import multiprocessing
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, data_precision, data_region, load_query
from src.spatial_index.learned.zm_index import Array
from src.spatial_index.proposed.slbrin import SLBRIN, HistoryRange, NN, valid_position_funcs, range_position_funcs
from src.ts_predict import TimeSeriesModel
from src.utils.common_utils import binary_search_less_max_duplicate, binary_search_less_max, merge_sorted_list, \
    biased_search_duplicate, binary_search_duplicate, biased_search_almost

PAGE_SIZE = 4096
HR_SIZE = 8 + 1 + 2 + 4 + 1  # 16
CR_SIZE = 8 * 4 + 2 + 1  # 35
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class USLBRIN(SLBRIN):
    """
    动态空间块范围学习型索引（Updatable Spatial Learned Block Range Index，USLBRIN）
    1. 基本思路：结合SBRIN和TSUM
    """

    def __init__(self, model_path=None):
        super().__init__(model_path)
        # for update
        self.history_ranges_append = None
        self.start_time = 0
        self.time_id = 0
        self.time_interval = 0  # T
        self.lag = 0  # l
        self.predict_step = 0  # f
        self.cdf_width = 0  # c
        self.child_length = 0  # bs
        self.cdf_model = None  # MF
        self.max_key_model = None  # Mn
        self.is_init = False
        self.threshold_err = 1
        self.threshold_err_cdf = 1
        self.threshold_err_max_key = 1
        # for compute
        self.is_retrain = True
        self.time_retrain = -1
        self.thread_retrain = 1
        self.is_save = True
        self.is_retrain_delta = True
        self.time_retrain_delta = -1
        self.thread_retrain_delta = 1
        self.is_save_delta = True
        self.insert_time = 0
        self.insert_io = 0
        self.last_insert_time = 0
        self.last_insert_io = 0

    def init_hr_append(self, hr, data):
        # create the old_cdfs and old_max_keys for delta_model
        key_interval = hr.value_diff / self.cdf_width
        key_list = [int(hr.value + k * key_interval) for k in range(self.cdf_width)]
        # TODO: UNIFORM数据集中time_id可能为float
        old_cdfs = [[] for k in range(int(self.time_id))]
        for tmp in data:
            old_cdfs[(tmp[3] - self.start_time) // self.time_interval].append(tmp[2])
        old_max_keys = [max(len(cdf) - 1, 0) for cdf in old_cdfs]
        # for empty and head old_cdfs, remove them
        l = 0
        while l < self.time_id and len(old_cdfs[l]) == 0:
            l += 1
        old_cdfs = old_cdfs[l:]
        old_max_keys = old_max_keys[l:]
        for k in range(len(old_cdfs)):
            cdf = old_cdfs[k]
            if cdf:  # for non-empty old_cdfs, create by data
                old_cdfs[k] = self.build_cdf(cdf, key_list)
            else:  # for empty and non-head old_cdfs, copy from their previous
                old_cdfs[k] = old_cdfs[k - 1]
        # plot_ts(cdfs)
        delta_model = TimeSeriesModel(key_list, self.model_path,
                                      old_cdfs, self.cdf_model,
                                      old_max_keys, self.max_key_model, 0)
        delta_model.build(self.lag, self.predict_step, self.cdf_width)
        # 2. change delta_index from [] into [[]]
        delta_index = [Array(self.child_length)
                       for i in range(delta_model.max_keys[delta_model.time_id] + 1)]
        return HistoryRangeAppend(delta_index, delta_model)

    def build_append(self, time_interval, start_time, end_time,
                     lag, predict_step, cdf_width, child_length, cdf_model, max_key_model,
                     is_init, threshold_err, threshold_err_cdf, threshold_err_max_key,
                     is_retrain, time_retrain, thread_retrain, is_save,
                     is_retrain_delta, time_retrain_delta, thread_retrain_delta, is_save_delta,
                     is_build=True):
        """
        1. create delta_model with ts_model
        2. change delta_index from [] into [[]]
        """
        self.start_time = start_time
        self.time_id = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
        self.lag = lag
        self.predict_step = predict_step
        self.cdf_width = cdf_width
        self.child_length = child_length
        self.cdf_model = cdf_model
        self.max_key_model = max_key_model
        self.is_init = is_init
        self.threshold_err = threshold_err
        self.threshold_err_cdf = threshold_err_cdf
        self.threshold_err_max_key = threshold_err_max_key
        self.is_retrain = is_retrain
        self.time_retrain = time_retrain
        self.thread_retrain = thread_retrain
        self.is_save = is_save
        self.is_retrain_delta = is_retrain_delta
        self.time_retrain_delta = time_retrain_delta
        self.thread_retrain_delta = thread_retrain_delta
        self.is_save_delta = is_save_delta
        retrain_delta_model_mae1 = 0
        retrain_delta_model_mae2 = 0
        self.history_ranges_append = []
        if is_build:
            # 1. create delta_model with ts_model
            for i in range(self.meta.last_hr + 1):
                s = time.time()
                hr_append = self.init_hr_append(self.history_ranges[i], self.index_entries[i])
                retrain_delta_model_mae1 += hr_append.delta_model.cdf_verify_mae
                retrain_delta_model_mae2 += hr_append.delta_model.max_key_verify_mae
                self.history_ranges_append.append(hr_append)
                self.logging.info("%s: %s" % (i, time.time() - s))
        else:
            delta_models = np.load(os.path.join(self.model_path, 'delta_models.npy'), allow_pickle=True)
            for i in range(self.meta.last_hr + 1):
                delta_model = delta_models[i]
                delta_index = [Array(self.child_length) for i in range(delta_model.max_keys[delta_model.time_id] + 1)]
                self.history_ranges_append.append(HistoryRangeAppend(delta_index, delta_model))
        self.logging.info("Build delta model cdf mae: %s" % (retrain_delta_model_mae1 / (self.meta.last_hr + 1)))
        self.logging.info("Build delta model max_key mae: %s" % (retrain_delta_model_mae2 / (self.meta.last_hr + 1)))

    def build_cdf(self, data, key_list):
        x_len = len(data)
        x_max_key = x_len - 1
        cdf = []
        p = 0
        for l in range(self.cdf_width):
            p = binary_search_less_max_duplicate(data, key_list[l], p, x_max_key)
            cdf.append(p / x_len)
        return cdf

    def get_delta_index_key(self, key, hr, hr_append):
        """
        get the delta_index list which contains the key
        """
        delta_model = hr_append.delta_model
        pos = (key - hr.value) / hr.value_diff * self.cdf_width
        pos_int = int(pos)
        if pos < 0:
            key = 0
        elif pos >= self.cdf_width - 1:  # if point is at the top of cdf(1.0), insert into the tail of delta_index
            key = delta_model.max_keys[delta_model.time_id]
        else:
            cdf = delta_model.cdfs[delta_model.time_id]
            left_p, right_p = cdf[pos_int: pos_int + 2]
            key = int((left_p + (right_p - left_p) * (pos - pos_int)) * delta_model.max_keys[delta_model.time_id])
        return key

    def insert_single(self, point):
        gh = self.meta.geohash.encode(point[0], point[1])
        point = (point[0], point[1], gh, point[2], point[3])
        hr_key = self.binary_search_less_max(gh, 0, self.meta.last_hr)
        # 1. find and insert ie into the target list of delta_index
        hr = self.history_ranges[hr_key]
        hr_append = self.history_ranges_append[hr_key]
        hr_append.delta_model.data_len += 1
        tg_array = hr_append.delta_index[self.get_delta_index_key(gh, hr, hr_append)]
        tg_array.insert(binary_search_less_max(tg_array.index, 2, gh, 0, tg_array.max_key) + 1, point)
        # IO1: search key
        self.io_cost += math.ceil((tg_array.max_key + 1) / ITEMS_PER_PAGE)

    def insert(self, points):
        points = points.tolist()
        for point in points:
            cur_time = point[2]
            # update once the time of new point cross the time interval
            time_id = (cur_time - self.start_time) // self.time_interval
            if self.time_id < time_id:
                self.time_id = time_id
                self.update()
            start_time = time.time()
            io_cost = self.io_cost
            self.insert_single(point)
            self.insert_time += time.time() - start_time
            self.insert_io += self.io_cost - io_cost

    def update_hr(self, hr_key, points):
        """
        update hr by points
        """
        # merge cr data into hr data
        hr = self.history_ranges[hr_key]
        # quicksort->merge_sorted_array->sorted->binary_search and insert => 50:2:1:0.5
        hr_data = self.index_entries[hr_key]
        merge_sorted_list(hr_data, points)
        hr_number = len(hr_data)
        if hr_number > self.meta.threshold_number and hr.length < self.meta.threshold_length:
            # split hr
            return self.split_hr(hr, hr_key, hr_data) - 1
        else:
            # update hr metadata
            hr.number = hr_number
            hr.max_key = hr_number - 1
            old_err = hr.model.max_err - hr.model.min_err
            hr.update_error_range(hr_data)
            if hr.model.max_err - hr.model.min_err > self.threshold_err * old_err:
                hr.state = 1
            return 0

    def split_hr(self, hr, hr_key, hr_data):
        # 1. create child hrs, of which model is inherited from parent hr and update err by inherited index entries
        region_offset = pow(10, -self.meta.geohash.data_precision - 1)
        range_stack = [(hr.value, hr.length, len(hr_data), 0, hr.scope.up_right_more_region(region_offset), hr.model)]
        range_list = []
        while len(range_stack):
            cur = range_stack.pop(-1)
            if cur[2] > self.meta.threshold_number and cur[1] < self.meta.threshold_length:
                child_regions = cur[4].split()
                l_key = cur[3]
                r_key = cur[3] + cur[2] - 1
                tmp_l_key = l_key
                child_list = [None] * 4
                length = cur[1] + 2
                r_bound = cur[0]
                for i in range(4):
                    value = r_bound
                    r_bound = cur[0] + (i + 1 << self.meta.geohash.sum_bits - length)
                    tmp_r_key = binary_search_less_max(hr_data, 2, r_bound, tmp_l_key, r_key)
                    child_list[i] = (value, length, tmp_r_key - tmp_l_key + 1, tmp_l_key, child_regions[i], cur[5])
                    tmp_l_key = tmp_r_key + 1
                range_stack.extend(child_list[::-1])
            else:
                range_list.append(cur)
        child_len = len(range_list)
        child_ranges = []
        old_err = hr.model.max_err - hr.model.min_err
        for r in range_list:
            child_data = hr_data[r[3]:r[3] + r[2]]
            child_hr = HistoryRange(r[0], r[1], r[2], r[5], 0, r[4].up_right_less_region(region_offset),
                                    2 << self.meta.geohash.sum_bits - r[1] - 1)
            child_hr.update_error_range(child_data)
            if child_hr.model.max_err - child_hr.model.min_err > self.threshold_err * old_err:
                child_hr.state = 1
            chiid_hr_append = self.init_hr_append(child_hr, child_data)
            child_ranges.append([child_hr, chiid_hr_append, child_data])
        # 2. replace old hr and data
        del self.index_entries[hr_key]
        del self.history_ranges[hr_key]
        del self.history_ranges_append[hr_key]
        child_ranges.reverse()  # 倒序一下，有助于insert
        for child_range in child_ranges:
            self.history_ranges.insert(hr_key, child_range[0])
            self.history_ranges_append.insert(hr_key, child_range[1])
            self.index_entries.insert(hr_key, child_range[2])
        # 3. update meta
        self.meta.last_hr += child_len - 1
        return child_len

    def update(self):
        """
        update the whole index
        1. merge delta index into index
        2. update model
        3. update delta model
        """
        self.logging.info("Update time id: %s" % self.time_id)
        self.logging.info("Insert key time: %s" % (self.insert_time - self.last_insert_time))
        self.logging.info("Insert key io: %s" % (self.insert_io - self.last_insert_io))
        delta_model_mae1 = 0
        delta_model_mae2 = 0
        hr_num = self.meta.last_hr + 1
        for hr_append in self.history_ranges_append:
            data_len = hr_append.delta_model.data_len
            pre_data_len = hr_append.delta_model.max_keys[hr_append.delta_model.time_id] + 1
            if data_len:
                cur_mae1 = 0
                for tg_array in hr_append.delta_index:
                    cur_mae1 += abs((tg_array.max_key + 1) / data_len - 1 / pre_data_len)
                cur_mae1 = cur_mae1 / pre_data_len
                delta_model_mae1 += cur_mae1
                hr_append.delta_model.cdf_real_mae = cur_mae1
            cur_mae2 = abs(data_len - pre_data_len)
            delta_model_mae2 += cur_mae2
            hr_append.delta_model.max_key_real_mae = cur_mae2
        delta_model_mae1 = delta_model_mae1 / hr_num
        delta_model_mae2 = delta_model_mae2 / hr_num
        self.logging.info("Delta model cdf mae: %s" % delta_model_mae1)
        self.logging.info("Delta model max_key mae: %s" % delta_model_mae2)
        self.last_insert_time = self.insert_time
        self.last_insert_io = self.insert_io
        # 1. merge delta index into index
        cdfs = []
        max_keys = []
        start_io = self.io_cost
        start_time = time.time()
        offset = 0  # update_hr中若出现split_hr，会导致后续hr_key向后偏移，因此用offset来记录偏移量
        for i in range(0, hr_num):
            hr_append = self.history_ranges_append[i + offset]
            if hr_append.delta_model.data_len:
                # merge data
                hr_append.delta_model.data_len = 0
                delta_index = []
                for tmp in hr_append.delta_index:
                    delta_index.extend(tmp.index[:tmp.max_key + 1])
                tmp = self.update_hr(i + offset, delta_index)
                offset += tmp
                if tmp:
                    cdfs.extend([None] * (tmp + 1))
                    max_keys.extend([None] * (tmp + 1))
                else:
                    cur_cdf = self.build_cdf([data[2] for data in delta_index], hr_append.delta_model.key_list)
                    cur_max_key = len(delta_index) - 1
                    cdfs.append(cur_cdf)
                    max_keys.append(cur_max_key)
                # IO1: merge data
                self.io_cost += math.ceil(len(self.index_entries[i]) / ITEMS_PER_PAGE)
            else:
                cdfs.append(None)
                max_keys.append(None)
        hr_num += offset
        self.logging.info("Merge data time: %s" % (time.time() - start_time))
        self.logging.info("Merge data io: %s" % (self.io_cost - start_io))
        # 2. update model
        if self.is_retrain and self.time_id > self.time_retrain:
            retrain_model_num = 0
            retrain_model_epoch = 0
            start_time = time.time()
            pool = multiprocessing.Pool(processes=self.thread_retrain)
            mp_dict = multiprocessing.Manager().dict()
            for i in range(0, hr_num):
                hr = self.history_ranges[i]
                if hr.state:
                    pool.apply_async(retrain_model,
                                     (self.model_path, i, self.index_entries[i], hr,
                                      self.weight, self.cores, self.train_step, self.batch_num, self.learning_rate,
                                      self.is_init, mp_dict))
            pool.close()
            pool.join()
            for (key, value) in mp_dict.items():
                value[0].state = 0
                self.history_ranges[key] = value[0]
                retrain_model_num += value[1]
                retrain_model_epoch += value[2]
            self.logging.info("Retrain model num: %s" % retrain_model_num)
            self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)
            self.logging.info("Retrain model time: %s" % (time.time() - start_time))
            self.logging.info("Retrain model io: %s" % (self.io_cost - start_io))
        else:
            time_model_path = os.path.join(self.model_path, "../uslbrin_time_model", str(self.time_id),
                                           'models_%s_%s.npy' % (self.is_init, self.threshold_err))
            models = np.load(time_model_path, allow_pickle=True)
            for i in range(0, hr_num):
                self.history_ranges[i].state = 0
                self.history_ranges[i].model = models[i]
        if self.is_save:
            time_model_path = os.path.join(self.model_path, "../uslbrin_time_model", str(self.time_id))
            if os.path.exists(time_model_path) is False:
                os.makedirs(time_model_path)
            models = [hr.model for hr in self.history_ranges]
            np.save(os.path.join(time_model_path, 'models_%s_%s.npy' % (self.is_init, self.threshold_err)), models)
        # 3. update delta model
        retrain_delta_model_num1 = 0
        retrain_delta_model_num2 = 0
        retrain_delta_model_mae1 = 0
        retrain_delta_model_mae2 = 0
        retrain_delta_model_time = 0
        retrain_delta_model_io = 0
        if self.is_retrain_delta and self.time_id > self.time_retrain_delta:
            start_time = time.time()
            for i in range(0, hr_num):
                hr_append = self.history_ranges_append[i]
                if cdfs[i]:
                    num_cdf, num_max_key = hr_append.delta_model.update(cdfs[i], max_keys[i], self.lag,
                                                                        self.predict_step,
                                                                        self.cdf_width, self.threshold_err_cdf,
                                                                        self.threshold_err_max_key)
                    hr_append.delta_index = [Array(self.child_length) for i in range(
                        hr_append.delta_model.max_keys[hr_append.delta_model.time_id] + 1)]
                    retrain_delta_model_num1 += num_cdf
                    retrain_delta_model_num2 += num_max_key
                    if num_cdf or num_max_key:
                        retrain_delta_model_io += len(hr_append.delta_model.max_keys)
                    else:
                        retrain_delta_model_io += 1
                retrain_delta_model_mae1 += hr_append.delta_model.cdf_verify_mae
                retrain_delta_model_mae2 += hr_append.delta_model.max_key_verify_mae
            retrain_delta_model_time = (time.time() - start_time)
            retrain_delta_model_io = retrain_delta_model_io * (self.cdf_width + 1) * 8 / PAGE_SIZE
            retrain_delta_model_mae1 = retrain_delta_model_mae1 / hr_num
            retrain_delta_model_mae2 = retrain_delta_model_mae2 / hr_num
        else:
            time_model_path = os.path.join(
                self.model_path, "../uslbrin_time_model", str(self.time_id),
                'delta_models_%s_%s.npy' % (self.threshold_err_cdf, self.threshold_err_max_key))
            delta_models = np.load(time_model_path, allow_pickle=True)
            for i in range(0, hr_num):
                if delta_models[i]:
                    retrain_delta_model_mae1 += delta_models[i].cdf_verify_mae
                    retrain_delta_model_mae2 += delta_models[i].max_key_verify_mae
                    self.history_ranges_append[i].delta_model = delta_models[i]
                    self.history_ranges_append[i].delta_index = [Array(self.child_length) for i in range(
                        delta_models[i].max_keys[delta_models[i].time_id] + 1)]
            retrain_delta_model_mae1 = retrain_delta_model_mae1 / hr_num
            retrain_delta_model_mae2 = retrain_delta_model_mae2 / hr_num
        if self.is_save_delta:
            time_model_path = os.path.join(self.model_path, "../uslbrin_time_model", str(self.time_id))
            if os.path.exists(time_model_path) is False:
                os.makedirs(time_model_path)
            delta_models = [hr_append.delta_model for hr_append in self.history_ranges_append]
            np.save(os.path.join(time_model_path, 'delta_models_%s_%s.npy' %
                                 (self.threshold_err_cdf, self.threshold_err_max_key)), delta_models)
        index_len = 0
        for i in range(0, hr_num):
            index_len += len(self.index_entries[i]) + len(self.history_ranges_append[i].delta_index) * self.child_length
        self.logging.info("Retrain delta model cdf num: %s" % retrain_delta_model_num1)
        self.logging.info("Retrain delta model max_key num: %s" % retrain_delta_model_num2)
        self.logging.info("Retrain delta model cdf mae: %s" % retrain_delta_model_mae1)
        self.logging.info("Retrain delta model max_key mae: %s" % retrain_delta_model_mae2)
        self.logging.info("Retrain delta model time: %s" % retrain_delta_model_time)
        self.logging.info("Retrain delta model io: %s" % retrain_delta_model_io)
        self.logging.info("Retrain index entry size: %s" % (index_len * ITEM_SIZE))
        self.logging.info("Retrain error bound: %s" % self.model_err())

    def point_query_single(self, point):
        """
        1. compute geohash from x/y of points
        2. find hr within geohash by slbrin.point_query
        3. predict by leaf model
        4. biased search in scope [pre - max_err, pre + min_err]
        """
        # 1. compute geohash from x/y of point
        gh = self.meta.geohash.encode(point[0], point[1])
        # 2. find hr within geohash by slbrin.point_query
        hr_key = self.point_query_hr(gh)
        hr = self.history_ranges[hr_key]
        if hr.number == 0:
            result = []
        else:
            # 3. predict by leaf model
            pre = hr.model_predict(gh)
            target_ies = self.index_entries[hr_key]
            # 4. biased search in scope [pre - max_err, pre + min_err]
            l_bound = max(pre - hr.model.max_err, 0)
            r_bound = min(pre - hr.model.min_err, hr.max_key)
            self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
            result = [target_ies[key][4] for key in biased_search_duplicate(target_ies, 2, gh, pre, l_bound, r_bound)]
        hr_append = self.history_ranges_append[hr_key]
        tg_array = hr_append.delta_index[self.get_delta_index_key(gh, hr, hr_append)]
        result.extend([tg_array.index[key][4]
                       for key in binary_search_duplicate(tg_array.index, 2, gh, 0, tg_array.max_key)])
        self.io_cost += math.ceil((tg_array.max_key + 1) / ITEMS_PER_PAGE)
        return result

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
            hr_append = self.history_ranges_append[hr_key]
            position = hr_list[hr_key]
            hr_data = self.index_entries[hr_key]
            if position == 0:  # window contain hr
                result.extend([ie[4] for ie in hr_data])
                self.io_cost += math.ceil(len(hr_data) / ITEMS_PER_PAGE)
                delta_index_len = 0
                for child in hr_append.delta_index:
                    result.extend([ie[4] for ie in child.index[:child.max_key + 1]])
                    delta_index_len += child.max_key + 1
                self.io_cost += math.ceil(delta_index_len / ITEMS_PER_PAGE)
            else:
                # wrong child hr from range_by_int
                is_valid = valid_position_funcs[position](hr.scope, window)
                if not is_valid:
                    continue
                # if-elif-else->lambda, 30->4
                gh_new1, gh_new2, compare_func = range_position_funcs[position](hr.scope, window, gh1, gh2,
                                                                                self.meta.geohash)
                # 4 predict min_key/max_key by nn
                if gh_new1:
                    pre1 = hr.model_predict(gh_new1)
                    l_bound1 = max(pre1 - hr.model.max_err, 0)
                    r_bound1 = min(pre1 - hr.model.min_err, hr.max_key)
                    left_key = min(biased_search_almost(hr_data, 2, gh_new1, pre1, l_bound1, r_bound1))
                    left_key_append = self.get_delta_index_key(gh_new1, hr, hr_append)
                else:
                    l_bound1 = 0
                    left_key = 0
                    left_key_append = 0
                if gh_new2:
                    pre2 = hr.model_predict(gh_new2)
                    l_bound2 = max(pre2 - hr.model.max_err, 0)
                    r_bound2 = min(pre2 - hr.model.min_err, hr.max_key)
                    right_key = max(biased_search_almost(hr_data, 2, gh_new2, pre2, l_bound2, r_bound2)) + 1
                    right_key_append = self.get_delta_index_key(gh_new2, hr, hr_append) + 1
                else:
                    r_bound2 = hr.number
                    right_key = hr.number
                    right_key_append = len(hr_append.delta_index)
                # 5 filter all the point of scope[min_key/max_key] by range.contain(point)
                # 优化: region.contain->compare_func不同位置的点做不同的判断: 638->474mil
                result.extend([ie[4] for ie in hr_data[left_key:right_key] if compare_func(ie)])
                self.io_cost += math.ceil((r_bound2 - l_bound1) / ITEMS_PER_PAGE)
                delta_index_len = 0
                for child in hr_append.delta_index[left_key_append:right_key_append]:
                    result.extend([ie[4] for ie in child.index[:child.max_key + 1] if compare_func(ie)])
                    delta_index_len += child.max_key + 1
                self.io_cost += math.ceil(delta_index_len / ITEMS_PER_PAGE)
        return result

    def knn_query_single(self, knn):
        """
        1. get the nearest key of query point
        2. get the nn points to create range query window
        3. filter point by distance
        主要耗时间：knn_query_hr/nn predict/精确过滤: 6.1/30/40.5
        """
        x, y, k = knn
        k = int(k)
        # 1. get the nearest key of query point
        qp_g = self.meta.geohash.encode(x, y)
        qp_hr_key = self.point_query_hr(qp_g)
        qp_hr = self.history_ranges[qp_hr_key]
        qp_hr_data = self.index_entries[qp_hr_key]
        # if hr is empty, TODO
        if qp_hr.number == 0:
            return []
        # if model, qp_ie_key = point_query(geohash)
        else:
            pre = qp_hr.model_predict(qp_g)
            l_bound = max(pre - qp_hr.model.max_err, 0)
            r_bound = min(pre - qp_hr.model.min_err, qp_hr.max_key)
            qp_ie_key = biased_search_almost(qp_hr_data, 2, qp_g, pre, l_bound, r_bound)[0]
        # 2. get the n points to create range query window
        tp_ie_list = [qp_hr_data[qp_ie_key]]
        cur_ie_key = qp_ie_key + 1
        cur_hr_data = qp_hr_data
        cur_hr = qp_hr
        cur_hr_key = qp_hr_key
        i = k
        while i > 0:
            right_ie_len = cur_hr.number - cur_ie_key + 1
            if right_ie_len >= i:
                tp_ie_list.extend(cur_hr_data[cur_ie_key:cur_ie_key + i])
                break
            else:
                tp_ie_list.extend(cur_hr_data[cur_ie_key:])
                if cur_hr_key == self.meta.last_hr:
                    break
                i -= right_ie_len
                cur_hr_key += 1
                cur_hr_data = self.index_entries[cur_hr_key]
                cur_hr = self.history_ranges[cur_hr_key]
                cur_ie_key = 0
        cur_ie_key = qp_ie_key
        cur_hr_key = qp_hr_key
        cur_hr_data = qp_hr_data
        i = k
        while i > 0:
            left_ie_len = cur_ie_key
            if left_ie_len >= i:
                tp_ie_list.extend(cur_hr_data[cur_ie_key - i:cur_ie_key])
                break
            else:
                tp_ie_list.extend(cur_hr_data[cur_ie_key - left_ie_len:cur_ie_key])
                if cur_hr_key == 0:
                    break
                i -= left_ie_len
                cur_hr_key -= 1
                cur_hr_data = self.index_entries[cur_hr_key]
                cur_ie_key = self.history_ranges[cur_hr_key].number
        tp_list = sorted([[(tp_ie[0] - x) ** 2 + (tp_ie[1] - y) ** 2, tp_ie[4]] for tp_ie in tp_ie_list])[:k]
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
            hr_key = tp_window_hr[0]
            hr = self.history_ranges[hr_key]
            hr_append = self.history_ranges_append[hr_key]
            position = tp_window_hr[1]
            hr_data = self.index_entries[hr_key]
            if position == 0:  # window contain hr
                tmp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]] for ie in hr_data]
                self.io_cost += math.ceil(len(hr_data) / ITEMS_PER_PAGE)
                delta_index_len = 0
                for child in hr_append.delta_index:
                    tmp_list.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]]
                                     for ie in child.index[:child.max_key + 1]])
                    delta_index_len += child.max_key + 1
                self.io_cost += math.ceil(delta_index_len / ITEMS_PER_PAGE)
            else:
                # wrong child hr from range_by_int
                is_valid = valid_position_funcs[position](hr.scope, window)
                if not is_valid:
                    continue
                gh_new1, gh_new2, compare_func = range_position_funcs[tp_window_hr[1]](hr.scope, window, gh1, gh2,
                                                                                       self.meta.geohash)
                if gh_new1:
                    pre1 = hr.model_predict(gh_new1)
                    l_bound1 = max(pre1 - hr.model.max_err, 0)
                    r_bound1 = min(pre1 - hr.model.min_err, hr.max_key)
                    left_key = min(biased_search_almost(hr_data, 2, gh_new1, pre1, l_bound1, r_bound1))
                    left_key_append = self.get_delta_index_key(gh_new1, hr, hr_append)
                else:
                    l_bound1 = 0
                    left_key = 0
                    left_key_append = 0
                if gh_new2:
                    pre2 = hr.model_predict(gh_new2)
                    l_bound2 = max(pre2 - hr.model.max_err, 0)
                    r_bound2 = min(pre2 - hr.model.min_err, hr.max_key)
                    right_key = max(biased_search_almost(hr_data, 2, gh_new2, pre2, l_bound2, r_bound2)) + 1
                    right_key_append = self.get_delta_index_key(gh_new2, hr, hr_append) + 1
                else:
                    r_bound2 = hr.number
                    right_key = hr.number
                    right_key_append = len(hr_append.delta_index)
                # 3. filter point by distance
                tmp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]]
                            for ie in hr_data[left_key:right_key] if compare_func(ie)]
                self.io_cost += math.ceil((r_bound2 - l_bound1) / ITEMS_PER_PAGE)
                delta_index_len = 0
                for child in hr_append.delta_index[left_key_append:right_key_append]:
                    tmp_list.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]]
                                     for ie in child.index[:child.max_key + 1] if compare_func(ie)])
                    delta_index_len += child.max_key + 1
                self.io_cost += math.ceil(delta_index_len / ITEMS_PER_PAGE)
            if len(tmp_list) > 0:
                tp_list.extend(tmp_list)
                tp_list = sorted(tp_list)[:k]
                max_dist = tp_list[-1][0]
        return [tp[1] for tp in tp_list]

    def save(self):
        super(USLBRIN, self).save()
        meta_append = np.array((self.start_time, self.time_id, self.time_interval,
                                self.lag, self.predict_step, self.cdf_width,
                                self.is_init, self.threshold_err, self.threshold_err_cdf, self.threshold_err_max_key),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'),
                                      ("3", 'i1'), ("4", 'i1'), ("5", 'i2'),
                                      ("6", 'i1'), ("7", 'f8'), ("8", 'f8'), ("9", 'f8')])
        np.save(os.path.join(self.model_path, 'meta_append.npy'), meta_append)
        compute = np.array((self.is_retrain, self.time_retrain, self.thread_retrain, self.is_save,
                            self.is_retrain_delta, self.time_retrain_delta, self.thread_retrain_delta,
                            self.is_save_delta),
                           dtype=[("0", 'i1'), ("1", 'i2'), ("2", 'i1'), ("3", 'i1'),
                                  ("4", 'i1'), ("5", 'i2'), ("6", 'i1'), ("7", 'i1')])
        np.save(os.path.join(self.model_path, 'compute.npy'), compute)
        delta_models = np.array([hr.delta_model for hr in self.history_ranges_append])
        np.save(os.path.join(self.model_path, 'delta_models.npy'), delta_models)
        delta_indexes = []
        delta_index_lens = []
        for hr in self.history_ranges_append:
            for array in hr.delta_index:
                delta_indexes.extend(array.index)
                delta_index_lens.append(array.size)
                delta_index_lens.append(array.max_key)
        np.save(os.path.join(self.model_path, 'delta_indexes.npy'),
                np.array(delta_indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'delta_index_lens.npy'), delta_index_lens)

    def load(self):
        super(USLBRIN, self).load()
        meta_append = np.load(os.path.join(self.model_path, 'meta_append.npy'), allow_pickle=True).item()
        self.start_time = meta_append[0]
        self.time_id = meta_append[1]
        self.time_interval = meta_append[2]
        self.initial_length = meta_append[3]
        self.is_init = bool(meta_append[4])
        self.threshold_err = meta_append[5]
        compute = np.load(os.path.join(self.model_path, 'compute.npy'), allow_pickle=True).item()
        self.is_retrain = bool(compute[0])
        self.time_retrain = compute[1]
        self.thread_retrain = compute[2]
        self.is_save = bool(compute[3])
        delta_indexes = np.load(os.path.join(self.model_path, 'delta_indexes.npy'), allow_pickle=True).tolist()
        delta_index_lens = np.load(os.path.join(self.model_path, 'delta_index_lens.npy'), allow_pickle=True).tolist()
        delta_models = np.load(os.path.join(self.model_path, 'delta_models.npy'), allow_pickle=True)
        self.history_ranges_append = []
        for i in range(self.meta.last_hr + 1):
            delta_index_cur = 0
            delta_index_len_cur = 0
            delta_model = delta_models[i]
            cur_max_key = delta_model.max_keys[delta_model.time_id]
            delta_index = []
            for j in range(cur_max_key + 1):
                size = delta_index_lens[delta_index_len_cur]
                max_key = delta_index_lens[delta_index_len_cur + 1]
                index = delta_indexes[delta_index_cur:delta_index_cur + size]
                delta_index_len_cur += 2
                delta_index_cur += size
                delta_index.append(Array(size, max_key, index))
            self.history_ranges_append.append(HistoryRangeAppend(delta_index, delta_model))

    def size(self):
        """
        structure_size += meta_append.npy
        ie_size
        """
        structure_size, ie_size = super(USLBRIN, self).size()
        ie_size += sum([child.size
                        for hr_append in self.history_ranges_append
                        for child in hr_append.delta_index]) * ITEM_SIZE
        structure_size += os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128 + \
                          os.path.getsize(os.path.join(self.model_path, "delta_models.npy")) - 128
        return structure_size, ie_size

    def model_err(self):
        return sum([(hr.model.max_err - hr.model.min_err) for hr in self.history_ranges]) / (self.meta.last_hr + 1)


class HistoryRangeAppend:
    def __init__(self, delta_index, delta_model):
        self.delta_index = delta_index
        self.delta_model = delta_model


def retrain_model(model_path, model_key, inputs, hr,
                  weight, cores, train_step, batch_num, learning_rate,
                  is_init, mp_dict):
    inputs = [data[2] for data in inputs]
    inputs.insert(0, hr.value)
    inputs.append(hr.value + hr.value_diff)
    inputs_num = hr.number + 2
    labels = list(range(0, inputs_num))
    batch_size = 2 ** math.ceil(math.log(inputs_num / batch_num, 2))
    if batch_size < 1:
        batch_size = 1
    tmp_index = NN(model_path, model_key, inputs, labels, True, weight,
                   cores, train_step, batch_size, learning_rate, False, None, None)
    if is_init:
        tmp_index.build_simple(hr.model.matrices if hr.model else None)
    else:
        tmp_index.build_simple(None)
    hr.model.matrices = tmp_index.get_matrices()
    hr.model.min_err = math.floor(tmp_index.min_err)
    hr.model.max_err = math.ceil(tmp_index.max_err)
    mp_dict[model_key] = (hr, 1, tmp_index.get_epochs())


def main():
    load_index_from_json = True
    load_index_from_json2 = True
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/uslbrin_10w/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = USLBRIN(model_path=model_path)
    index_name = index.name
    if load_index_from_json:
        super(USLBRIN, index).load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    threshold_number=1000,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    threshold_err=1,
                    threshold_summary=1000,
                    threshold_merge=5,
                    is_new=False,
                    is_simple=False,
                    weight=1,
                    core=[1, 128],
                    train_step=5000,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=1,
                    thread_pool_size=3)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    if load_index_from_json2:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        index.build_append(time_interval=60 * 60 * 24,
                           start_time=1356998400,
                           end_time=1359676799,
                           lag=7,
                           predict_step=3,
                           cdf_width=100,
                           child_length=ITEMS_PER_PAGE,
                           cdf_model="var",
                           max_key_model="es",
                           is_init=True,
                           threshold_err=5,
                           threshold_err_cdf=2,
                           threshold_err_max_key=2,
                           is_retrain=False,
                           time_retrain=-1,
                           thread_retrain=3,
                           is_save=True,
                           is_retrain_delta=True,
                           time_retrain_delta=-1,
                           thread_retrain_delta=3,
                           is_save_delta=False,
                           is_build=False)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    io_cost = 0
    model_num = index.meta.last_hr + 1
    logging.info("Model num: %s" % model_num)
    model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
    model_precisions_avg = sum(model_precisions) / model_num
    logging.info("Error bound: %s" % model_precisions_avg)
    point_query_list = load_query(data_distribution, 0).tolist()
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
