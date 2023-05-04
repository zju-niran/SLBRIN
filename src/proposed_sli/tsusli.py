import logging
import math
import multiprocessing
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, data_precision, data_region, load_query
from src.utils.common_utils import biased_search_duplicate, binary_search_less_max, binary_search_duplicate, \
    Region, binary_search_less_max_duplicate
from src.utils.geohash_utils import Geohash
from src.sli.zm_index import Node, Array
from src.proposed_sli.slibs import SLIBS
from src.proposed_sli.dtusli import retrain_model
from src.ts_predict import TimeSeriesModel

PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class TSUSLI(SLIBS):
    """
    时空预测学习型索引（Tempo-Spatial updatable Spatial Learned Index，TSUSLI）
    1. 基本思路：提出时空预测更新方法（Tempo-Spatial predicted Update Method，TSUM），应用于SLIBS
    2. TSUM的间接预测：
    2.1. 空间分布分解为排列F（cdf）和密度n（max_key），分别用时空序列预测和时间序列预测完成预测，
    2.2. 将预测结果组合未来空间分布，改造增量区的数据结构，提供类似哈希索引的方法来检索其上的增量数据
    """
    def __init__(self, model_path=None):
        super().__init__(model_path)
        # for update
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
        if is_build:
            # 1. create delta_model with ts_model
            for j in range(self.stages[-1]):
                # index_lens = [(j, len(self.rmi[-1][j].index)) for j in range(self.stages[-1])]
                # index_lens.sort(key=lambda x: x[-1])
                # max_len_index = index_lens[-1][0]
                # for j in [max_len_index]:
                s = time.time()
                node = self.rmi[-1][j]
                # create the old_cdfs and old_max_keys for delta_model
                min_key = node.model.input_min
                max_key = node.model.input_max
                key_interval = (max_key - min_key) / cdf_width
                key_list = [int(min_key + k * key_interval) for k in range(cdf_width)]
                old_cdfs = [[] for k in range(self.time_id)]
                for data in node.index:
                    old_cdfs[(data[3] - self.start_time) // self.time_interval].append(data[2])
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
                node.delta_model = TimeSeriesModel(key_list, self.model_path,
                                                   old_cdfs, self.cdf_model,
                                                   old_max_keys, self.max_key_model, 0)
                node.delta_model.build(lag, predict_step, cdf_width)
                retrain_delta_model_mae1 += node.delta_model.cdf_verify_mae
                retrain_delta_model_mae2 += node.delta_model.max_key_verify_mae
                self.logging.info("%s: %s" % (j, time.time() - s))
                # 2. change delta_index from [] into [[]]
                node.delta_index = [Array(self.child_length)
                                    for i in range(node.delta_model.max_keys[node.delta_model.time_id] + 1)]
        else:
            delta_models = np.load(os.path.join(self.model_path, 'delta_models.npy'), allow_pickle=True)[
                           -self.stages[-1]:]
            for j in range(self.stages[-1]):
                self.rmi[-1][j].delta_model = delta_models[j]
                self.rmi[-1][j].delta_index = [Array(self.child_length)
                                               for i in range(delta_models[j].max_keys[delta_models[j].time_id] + 1)]
        self.logging.info("Build delta model cdf mae: %s" % (retrain_delta_model_mae1 / self.stages[-1]))
        self.logging.info("Build delta model max_key mae: %s" % (retrain_delta_model_mae2 / self.stages[-1]))

    def build_cdf(self, data, key_list):
        x_len = len(data)
        x_max_key = x_len - 1
        cdf = []
        p = 0
        for l in range(self.cdf_width):
            p = binary_search_less_max_duplicate(data, key_list[l], p, x_max_key)
            cdf.append(p / x_len)
        return cdf

    def insert_single(self, point):
        """
        different from zm_index
        1. find and insert ie into the target list of delta_index
        """
        gh = self.geohash.encode(point[0], point[1])
        point = (point[0], point[1], gh, point[2], point[3])
        leaf_node = self.rmi[-1][self.get_leaf_node(gh)]
        # 1. find and insert ie into the target list of delta_index
        leaf_node.delta_model.data_len += 1
        tg_array = leaf_node.delta_index[self.get_delta_index_key(gh, leaf_node)]
        tg_array.insert(binary_search_less_max(tg_array.index, 2, gh, 0, tg_array.max_key) + 1, point)
        # IO1: search key
        self.io_cost += math.ceil((tg_array.max_key + 1) / ITEMS_PER_PAGE)

    def insert(self, points):
        """
        different from zm_index
        1. update once the time of new point cross the time interval
        """
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
        for leaf_node in self.rmi[-1]:
            data_len = leaf_node.delta_model.data_len
            pre_data_len = leaf_node.delta_model.max_keys[leaf_node.delta_model.time_id] + 1
            if data_len:
                cur_mae1 = 0
                for tg_array in leaf_node.delta_index:
                    cur_mae1 += abs((tg_array.max_key + 1) / data_len - 1 / pre_data_len)
                cur_mae1 = cur_mae1 / pre_data_len
                delta_model_mae1 += cur_mae1
                leaf_node.delta_model.cdf_real_mae = cur_mae1
            cur_mae2 = abs(data_len - pre_data_len)
            delta_model_mae2 += cur_mae2
            leaf_node.delta_model.max_key_real_mae = cur_mae2
        delta_model_mae1 = delta_model_mae1 / self.stages[-1]
        delta_model_mae2 = delta_model_mae2 / self.stages[-1]
        self.logging.info("Delta model cdf mae: %s" % delta_model_mae1)
        self.logging.info("Delta model max_key mae: %s" % delta_model_mae2)
        self.last_insert_time = self.insert_time
        self.last_insert_io = self.insert_io
        # 1. merge delta index into index
        update_list = [0] * self.stages[-1]
        start_io = self.io_cost
        start_time = time.time()
        for j in range(0, self.stages[-1]):
            leaf_node = self.rmi[-1][j]
            if leaf_node.delta_model.data_len:
                leaf_node.delta_model.data_len = 0
                delta_index = []
                for tmp in leaf_node.delta_index:
                    delta_index.extend(tmp.index[:tmp.max_key + 1])
                update_list[j] = delta_index
                if leaf_node.index:
                    leaf_node.index.extend(delta_index)
                    leaf_node.index.sort(key=lambda x: x[2])  # 优化：有序数组合并->sorted:2.5->1
                else:
                    leaf_node.index = delta_index
                # IO1: merge data
                self.io_cost += math.ceil(len(leaf_node.index) / ITEMS_PER_PAGE)
        self.logging.info("Merge data time: %s" % (time.time() - start_time))
        self.logging.info("Merge data io: %s" % (self.io_cost - start_io))
        # 2. update model
        if self.is_retrain and self.time_id > self.time_retrain:
            retrain_model_num = 0
            retrain_model_epoch = 0
            start_time = time.time()
            pool = multiprocessing.Pool(processes=self.thread_retrain)
            mp_dict = multiprocessing.Manager().dict()
            for j in range(0, self.stages[-1]):
                if update_list[j]:
                    leaf_node = self.rmi[-1][j]
                    pool.apply_async(retrain_model,
                                     (self.model_path, j, leaf_node.index, leaf_node.model,
                                      self.weight, self.cores, self.train_step, self.batch_num, self.learning_rate,
                                      self.is_init, self.threshold_err, mp_dict))
            pool.close()
            pool.join()
            for (key, value) in mp_dict.items():
                self.rmi[-1][key].model = value[0]
                retrain_model_num += value[1]
                retrain_model_epoch += value[2]
            self.logging.info("Retrain model num: %s" % retrain_model_num)
            self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)
            self.logging.info("Retrain model time: %s" % (time.time() - start_time))
            self.logging.info("Retrain model io: %s" % (self.io_cost - start_io))
        else:
            time_model_path = os.path.join(self.model_path, "../sli_time_model_model", str(self.time_id), 'models.npy')
            models = np.load(time_model_path, allow_pickle=True)
            model_cur = 0
            for i in range(len(self.stages)):
                for j in range(self.stages[i]):
                    self.rmi[i][j].model = models[model_cur]
                    model_cur += 1
        if self.is_save:
            time_model_path = os.path.join(self.model_path, "../sli_time_model_model", str(self.time_id))
            if os.path.exists(time_model_path) is False:
                os.makedirs(time_model_path)
            models = []
            for stage in self.rmi:
                models.extend([node.model for node in stage])
            np.save(os.path.join(time_model_path, 'models.npy'), models)
        # 3. update delta model
        retrain_delta_model_num1 = 0
        retrain_delta_model_num2 = 0
        retrain_delta_model_mae1 = 0
        retrain_delta_model_mae2 = 0
        retrain_delta_model_time = 0
        retrain_delta_model_io = 0
        if self.is_retrain_delta and self.time_id > self.time_retrain_delta:
            start_time = time.time()
            pool = multiprocessing.Pool(processes=self.thread_retrain_delta)
            mp_dict = multiprocessing.Manager().dict()
            for j in range(0, self.stages[-1]):
                if update_list[j]:
                    leaf_node = self.rmi[-1][j]
                    cur_cdf = self.build_cdf([data[2] for data in update_list[j]], leaf_node.delta_model.key_list)
                    cur_max_key = len(update_list[j]) - 1
                    pool.apply_async(retrain_delta_model,
                                     (j, leaf_node.delta_model, cur_cdf, cur_max_key,
                                      self.lag, self.predict_step, self.cdf_width,
                                      self.threshold_err_cdf, self.threshold_err_max_key, mp_dict))
            pool.close()
            pool.join()
            for (key, value) in mp_dict.items():
                leaf_node = self.rmi[-1][key]
                leaf_node.delta_model = value[0]
                leaf_node.delta_index = [Array(self.child_length) for i in range(
                    leaf_node.delta_model.max_keys[leaf_node.delta_model.time_id] + 1)]
                retrain_delta_model_num1 += value[1]
                retrain_delta_model_num2 += value[2]
                if value[1]:
                    retrain_delta_model_io += len(leaf_node.delta_model.max_keys)
                else:
                    retrain_delta_model_io += 1
            for leaf_node in self.rmi[-1]:
                retrain_delta_model_mae1 += leaf_node.delta_model.cdf_verify_mae
                retrain_delta_model_mae2 += leaf_node.delta_model.max_key_verify_mae
            retrain_delta_model_time = (time.time() - start_time)
            retrain_delta_model_io = retrain_delta_model_io * (self.cdf_width + 1) * 8 / PAGE_SIZE
            retrain_delta_model_mae1 = retrain_delta_model_mae1 / self.stages[-1]
            retrain_delta_model_mae2 = retrain_delta_model_mae2 / self.stages[-1]
        else:
            time_model_path = os.path.join(
                self.model_path, "../sli_time_model_model", str(self.time_id), 'delta_models_%s_%s_%s_%s_%s.npy' % (
                    self.lag, self.predict_step, self.cdf_width, self.cdf_model, self.max_key_model))
            delta_models = np.load(time_model_path, allow_pickle=True)
            model_cur = 0
            for i in range(len(self.stages)):
                for j in range(self.stages[i]):
                    if delta_models[model_cur]:
                        retrain_delta_model_mae1 += delta_models[model_cur].cdf_verify_mae
                        retrain_delta_model_mae2 += delta_models[model_cur].max_key_verify_mae
                        self.rmi[i][j].delta_model = delta_models[model_cur]
                        self.rmi[i][j].delta_index = [Array(self.child_length) for i in range(
                            delta_models[model_cur].max_keys[delta_models[model_cur].time_id] + 1)]
                    model_cur += 1
            retrain_delta_model_mae1 = retrain_delta_model_mae1 / self.stages[-1]
            retrain_delta_model_mae2 = retrain_delta_model_mae2 / self.stages[-1]
        if self.is_save_delta:
            time_model_path = os.path.join(self.model_path, "../sli_time_model_model", str(self.time_id))
            if os.path.exists(time_model_path) is False:
                os.makedirs(time_model_path)
            delta_models = []
            for stage in self.rmi:
                delta_models.extend([node.delta_model for node in stage])
            np.save(os.path.join(time_model_path, 'delta_models_%s_%s_%s_%s_%s.npy' % (
                self.lag, self.predict_step, self.cdf_width, self.cdf_model, self.max_key_model)), delta_models)
        index_len = 0
        for leaf_node in self.rmi[-1]:
            index_len += len(leaf_node.index) + len(leaf_node.delta_index) * self.child_length
        self.logging.info("Retrain delta model cdf num: %s" % retrain_delta_model_num1)
        self.logging.info("Retrain delta model max_key num: %s" % retrain_delta_model_num2)
        self.logging.info("Retrain delta model cdf mae: %s" % retrain_delta_model_mae1)
        self.logging.info("Retrain delta model max_key mae: %s" % retrain_delta_model_mae2)
        self.logging.info("Retrain delta model time: %s" % retrain_delta_model_time)
        self.logging.info("Retrain delta model io: %s" % retrain_delta_model_io)
        self.logging.info("Retrain index entry size: %s" % (index_len * ITEM_SIZE))
        self.logging.info("Retrain error bound: %s" % self.model_err())

    def get_delta_index_key(self, key, leaf_node):
        """
        get the delta_index list which contains the key
        """
        model = leaf_node.model
        delta_model = leaf_node.delta_model
        pos = (key - model.input_min) / (model.input_max - model.input_min) * self.cdf_width
        pos_int = int(pos)
        if pos >= self.cdf_width - 1:  # if point is at the top of cdf(1.0), insert into the tail of delta_index
            key = delta_model.max_keys[delta_model.time_id]
        else:
            cdf = delta_model.cdfs[delta_model.time_id]
            left_p, right_p = cdf[pos_int: pos_int + 2]
            key = int((left_p + (right_p - left_p) * (pos - pos_int)) * delta_model.max_keys[delta_model.time_id])
        return key

    def point_query_single(self, point):
        """
        different from zm_index
        1. find the target list of delta_index which contains the target ie
        """
        gh = self.geohash.encode(point[0], point[1])
        leaf_node, _, pre, min_err, max_err = self.predict(gh)
        l_bound = max(pre - max_err, leaf_node.model.output_min)
        r_bound = min(pre - min_err, leaf_node.model.output_max)
        result = [leaf_node.index[key][4] for key in
                  biased_search_duplicate(leaf_node.index, 2, gh, pre, l_bound, r_bound)]
        self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
        # 1. find the target list of delta_index which contains the target ie
        if leaf_node.delta_model.data_len:
            tg_array = leaf_node.delta_index[self.get_delta_index_key(gh, leaf_node)]
            result.extend([tg_array.index[key][4]
                           for key in binary_search_duplicate(tg_array.index, 2, gh, 0, tg_array.max_key)])
            self.io_cost += math.ceil((tg_array.max_key + 1) / ITEMS_PER_PAGE)
        return result

    def range_query_single(self, window):
        """
        different from zm_index
        1. find the target list of delta_index which contains the target ie
        """
        gh1 = self.geohash.encode(window[2], window[0])
        gh2 = self.geohash.encode(window[3], window[1])
        leaf_node1, leaf_key1, pre1, min_err1, max_err1 = self.predict(gh1)
        l_bound1 = max(pre1 - max_err1, 0)
        r_bound1 = min(pre1 - min_err1, leaf_node1.model.output_max)
        left_key = biased_search_duplicate(leaf_node1.index, 2, gh1, pre1, l_bound1, r_bound1)
        left_key = l_bound1 if len(left_key) == 0 else min(left_key)
        leaf_node2, leaf_key2, pre2, min_err2, max_err2 = self.predict(gh2)
        l_bound2 = max(pre2 - max_err2, 0)
        r_bound2 = min(pre2 - min_err2, leaf_node2.model.output_max)
        right_key = biased_search_duplicate(leaf_node2.index, 2, gh2, pre2, l_bound2, r_bound2)
        right_key = r_bound2 if len(right_key) == 0 else max(right_key)
        # 1. find the target list of delta_index which contains the target ie
        io_index_len = 0
        io_delta_index_len = 0
        if leaf_key1 == leaf_key2:
            result = [ie[4] for ie in leaf_node1.index[left_key:right_key + 1]
                      if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
            io_index_len += r_bound2 - l_bound1
            # filter delta index
            if leaf_node1.delta_model.data_len:
                delta_index = leaf_node1.delta_index
                delta_left_key = self.get_delta_index_key(gh1, leaf_node1)
                delta_right_key = self.get_delta_index_key(gh2, leaf_node1) + 1
                result.extend([ie[4] for child in delta_index[delta_left_key:delta_right_key]
                               for ie in child.index[:child.max_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_delta_index_len += sum([child.max_key + 1 for child in delta_index[delta_left_key:delta_right_key]])
        else:
            io_index_len += len(leaf_node1.index) + r_bound2 - l_bound1
            result = [ie[4]
                      for ie in leaf_node1.index[left_key:]
                      if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
            if leaf_key2 - leaf_key1 > 1:
                result.extend([ie[4]
                               for leaf_key in range(leaf_key1 + 1, leaf_key2)
                               for ie in self.rmi[-1][leaf_key].index
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                for leaf_key in range(leaf_key1 + 1, leaf_key2):
                    io_index_len += len(self.rmi[-1][leaf_key].index)
            result.extend([ie[4]
                           for ie in leaf_node2.index[:right_key + 1]
                           if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
            # filter delta index
            if leaf_node1.delta_model.data_len:
                delta_index = leaf_node1.delta_index
                delta_left_key = self.get_delta_index_key(gh1, leaf_node1)
                result.extend([ie[4] for child in delta_index[delta_left_key:]
                               for ie in child.index[:child.max_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_delta_index_len += sum([child.max_key + 1 for child in delta_index[delta_left_key:]])
            if leaf_key2 - leaf_key1 > 1:
                result.extend([ie[4]
                               for leaf_key in range(leaf_key1 + 1, leaf_key2)
                               for child in self.rmi[-1][leaf_key].delta_index
                               for ie in child.index
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                for leaf_key in range(leaf_key1 + 1, leaf_key2):
                    io_delta_index_len += self.rmi[-1][leaf_key].delta_model.data_len
            if leaf_node2.delta_model.data_len:
                delta_index = leaf_node2.delta_index
                delta_right_key = self.get_delta_index_key(gh2, leaf_node2) + 1
                result.extend([ie[4] for child in delta_index[:delta_right_key]
                               for ie in child.index[:child.max_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_delta_index_len += sum([child.max_key + 1 for child in delta_index[:delta_right_key]])
        self.io_cost += math.ceil(io_index_len / ITEMS_PER_PAGE) + math.ceil(io_delta_index_len / ITEMS_PER_PAGE)
        return result

    def knn_query_single(self, knn):
        """
        different from zm_index
        1. find the target list of delta_index which contains the target ie
        """
        x, y, k = knn
        k = int(k)
        w = self.get_weight(self.geohash.encode(x, y))
        if w > 0:
            window_ratio = (k / self.max_key) ** 0.5 / w
        else:
            window_ratio = (k / self.max_key) ** 0.5
        window_radius = window_ratio * self.geohash.region_width / 2
        while True:
            window = [y - window_radius, y + window_radius, x - window_radius, x + window_radius]
            self.geohash.region.clip_region(window, self.geohash.data_precision)
            gh1 = self.geohash.encode(window[2], window[0])
            gh2 = self.geohash.encode(window[3], window[1])
            leaf_node1, leaf_key1, pre1, min_err1, max_err1 = self.predict(gh1)
            l_bound1 = max(pre1 - max_err1, 0)
            r_bound1 = min(pre1 - min_err1, leaf_node1.model.output_max)
            left_key = biased_search_duplicate(leaf_node1.index, 2, gh1, pre1, l_bound1, r_bound1)
            left_key = l_bound1 if len(left_key) == 0 else min(left_key)
            leaf_node2, leaf_key2, pre2, min_err2, max_err2 = self.predict(gh2)
            l_bound2 = max(pre2 - max_err2, 0)
            r_bound2 = min(pre2 - min_err2, leaf_node2.model.output_max)
            right_key = biased_search_duplicate(leaf_node2.index, 2, gh2, pre2, l_bound2, r_bound2)
            right_key = r_bound2 + 1 if len(right_key) == 0 else max(right_key) + 1
            io_index_len = 0
            io_delta_index_len = 0
            # 1. find the target list of delta_index which contains the target ie
            if leaf_key1 == leaf_key2:
                tp_list = [ie for ie in leaf_node1.index[left_key:right_key]
                           if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
                io_index_len += right_key - left_key
                # filter delta index
                if leaf_node1.delta_model.data_len:
                    delta_index = leaf_node1.delta_index
                    delta_left_key = self.get_delta_index_key(gh1, leaf_node1)
                    delta_right_key = self.get_delta_index_key(gh2, leaf_node1) + 1
                    tp_list.extend([ie for child in delta_index[delta_left_key:delta_right_key]
                                    for ie in child.index[:child.max_key + 1]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                    io_delta_index_len += sum(
                        [child.max_key + 1 for child in delta_index[delta_left_key:delta_right_key]])
            else:
                tp_list = [ie for ie in leaf_node1.index[left_key:]
                           if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
                io_index_len += len(leaf_node1.index) - left_key
                if leaf_node1.delta_model.data_len:
                    delta_index = leaf_node1.delta_index
                    delta_left_key = self.get_delta_index_key(gh1, leaf_node1)
                    tp_list.extend([ie for child in delta_index[delta_left_key:]
                                    for ie in child.index[:child.max_key + 1]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                    io_delta_index_len += sum([child.max_key + 1 for child in delta_index[delta_left_key:]])
                tp_list.extend([ie for ie in leaf_node2.index[:right_key]
                                if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_index_len += len(leaf_node1.index) + right_key
                if leaf_node2.delta_model.data_len:
                    delta_index = leaf_node2.delta_index
                    delta_right_key = self.get_delta_index_key(gh2, leaf_node2) + 1
                    tp_list.extend([ie for child in delta_index[:delta_right_key]
                                    for ie in child.index[:child.max_key + 1]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                    io_delta_index_len += sum([child.max_key + 1 for child in delta_index[:delta_right_key]])
                tp_list.extend([ie for leaf_key in range(leaf_key1 + 1, leaf_key2)
                                for ie in self.rmi[-1][leaf_key].index
                                if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_index_len += sum([len(self.rmi[-1][leaf_key].index) for leaf_key in range(leaf_key1 + 1, leaf_key2)])
                tp_list.extend([ie for leaf_key in range(leaf_key1 + 1, leaf_key2)
                                for child in self.rmi[-1][leaf_key].delta_index
                                for ie in child.index
                                if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_delta_index_len += sum(
                    [self.rmi[-1][leaf_key].delta_model.data_len for leaf_key in range(leaf_key1 + 1, leaf_key2)])
            if len(tp_list) < k:
                window_radius *= 2
            else:
                tp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[4]] for ie in tp_list]
                tp_list.sort()
                dst = tp_list[k - 1][0] ** 0.5
                if dst > window_radius:
                    window_radius = dst
                else:
                    break
        self.io_cost += math.ceil(io_index_len / ITEMS_PER_PAGE) + math.ceil(io_delta_index_len / ITEMS_PER_PAGE)
        return [tp[1] for tp in tp_list[:k]]

    def save(self):
        """
        different from zm_index
        1. the delta_index isn't list but a list of list
        """
        meta = np.array((self.geohash.data_precision,
                         self.geohash.region.bottom, self.geohash.region.up,
                         self.geohash.region.left, self.geohash.region.right,
                         self.weight, self.train_step, self.batch_num, self.learning_rate),
                        dtype=[("0", 'i4'),
                               ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8'),
                               ("5", 'f4'), ("6", 'i2'), ("7", 'i2'), ("8", 'f4')])
        np.save(os.path.join(self.model_path, 'meta.npy'), meta)
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
        np.save(os.path.join(self.model_path, 'stages.npy'), self.stages)
        np.save(os.path.join(self.model_path, 'cores.npy'), self.cores)
        models = []
        delta_models = []
        for stage in self.rmi:
            models.extend([node.model for node in stage])
            delta_models.extend([node.delta_model for node in stage])
        np.save(os.path.join(self.model_path, 'models.npy'), models)
        np.save(os.path.join(self.model_path, 'delta_models.npy'), delta_models)
        indexes = []
        index_lens = []
        delta_indexes = []
        delta_index_lens = []
        for node in self.rmi[-1]:
            indexes.extend(node.index)
            index_lens.append(len(node.index))
            for array in node.delta_index:
                delta_indexes.extend(array.index)
                delta_index_lens.append(array.size)
                delta_index_lens.append(array.max_key)
        np.save(os.path.join(self.model_path, 'indexes.npy'),
                np.array(indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'index_lens.npy'), index_lens)
        np.save(os.path.join(self.model_path, 'delta_indexes.npy'),
                np.array(delta_indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'delta_index_lens.npy'), delta_index_lens)

    def load(self):
        """
        different from zm_index
        1. the delta_index isn't [] but [[]]
        """
        meta = np.load(os.path.join(self.model_path, 'meta.npy'), allow_pickle=True).item()
        region = Region(meta[1], meta[2], meta[3], meta[4])
        self.geohash = Geohash.init_by_precision(data_precision=meta[0], region=region)
        self.stages = np.load(os.path.join(self.model_path, 'stages.npy'), allow_pickle=True).tolist()
        self.non_leaf_stage_len = len(self.stages) - 1
        self.cores = np.load(os.path.join(self.model_path, 'cores.npy'), allow_pickle=True).tolist()
        self.weight = meta[5]
        self.train_step = meta[6]
        self.batch_num = meta[7]
        self.learning_rate = meta[8]
        meta_append = np.load(os.path.join(self.model_path, 'meta_append.npy'), allow_pickle=True).item()
        self.start_time = meta_append[0]
        self.time_id = meta_append[1]
        self.time_interval = meta_append[2]
        self.lag = meta_append[3]
        self.predict_step = meta_append[4]
        self.cdf_width = meta_append[5]
        self.is_init = bool(meta_append[6])
        self.threshold_err = meta_append[7]
        self.threshold_err_cdf = meta_append[8]
        self.threshold_err_max_key = meta_append[9]
        compute = np.load(os.path.join(self.model_path, 'compute.npy'), allow_pickle=True).item()
        self.is_retrain = bool(compute[0])
        self.time_retrain = compute[1]
        self.thread_retrain = compute[2]
        self.is_save = bool(compute[3])
        self.is_retrain_delta = bool(compute[4])
        self.time_retrain_delta = compute[5]
        self.thread_retrain_delta = compute[6]
        self.is_save_delta = bool(compute[7])
        models = np.load(os.path.join(self.model_path, 'models.npy'), allow_pickle=True)
        indexes = np.load(os.path.join(self.model_path, 'indexes.npy'), allow_pickle=True).tolist()
        index_lens = np.load(os.path.join(self.model_path, 'index_lens.npy'), allow_pickle=True).tolist()
        delta_indexes = np.load(os.path.join(self.model_path, 'delta_indexes.npy'), allow_pickle=True).tolist()
        delta_index_lens = np.load(os.path.join(self.model_path, 'delta_index_lens.npy'), allow_pickle=True).tolist()
        delta_models = np.load(os.path.join(self.model_path, 'delta_models.npy'), allow_pickle=True)
        self.max_key = len(indexes)
        model_cur = 0
        self.rmi = []
        for i in range(len(self.stages)):
            if i < self.non_leaf_stage_len:
                models_tmp = models[model_cur:model_cur + self.stages[i]]
                delta_models_tmp = delta_models[model_cur:model_cur + self.stages[i]]
                self.rmi.append(
                    [Node(None, model, None, delta_model) for model, delta_model in zip(models_tmp, delta_models_tmp)])
                model_cur += self.stages[i]
            else:
                index_cur = 0
                delta_index_cur = 0
                delta_index_len_cur = 0
                leaf_nodes = []
                for j in range(self.stages[i]):
                    model = models[model_cur]
                    delta_model = delta_models[model_cur]
                    cur_max_key = delta_model.max_keys[delta_model.time_id]
                    delta_index = []
                    for i in range(cur_max_key + 1):
                        size = delta_index_lens[delta_index_len_cur]
                        max_key = delta_index_lens[delta_index_len_cur + 1]
                        index = delta_indexes[delta_index_cur:delta_index_cur + size]
                        delta_index_len_cur += 2
                        delta_index_cur += size
                        delta_index.append(Array(size, max_key, index))
                    index = indexes[index_cur: index_cur + index_lens[j]]
                    leaf_nodes.append(Node(index, model, delta_index, delta_model))
                    model_cur += 1
                    index_cur += index_lens[j]
                self.rmi.append(leaf_nodes)

    def size(self):
        """
        structure_size += meta_append.npy + delta_models.npy
        ie_size = index + delta_index
        """
        index_len = 0
        for leaf_node in self.rmi[-1]:
            index_len += len(leaf_node.index)
            for tg_array in leaf_node.delta_index:
                index_len += tg_array.size
        return os.path.getsize(os.path.join(self.model_path, "meta.npy")) - 128 - 64 * 2 + \
               os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "stages.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "cores.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "models.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "delta_models.npy")) - 128, \
               index_len * ITEM_SIZE


def retrain_delta_model(model_key, delta_model, cur_cdf, cur_max_key, lag, predict_step, cdf_width,
                        threshold_err_cdf, threshold_err_max_key, mp_dict):
    num1, num2 = delta_model.update(cur_cdf, cur_max_key, lag, predict_step, cdf_width,
                                    threshold_err_cdf, threshold_err_max_key)
    mp_dict[model_key] = delta_model, num1, num2


def main():
    load_index_from_json = True
    load_index_from_json2 = False
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/tsusli_10w/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = TSUSLI(model_path=model_path)
    index_name = index.name
    if load_index_from_json:
        super(TSUSLI, index).load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    is_new=False,
                    is_simple=False,
                    weight=1,
                    stages=[1, 100],
                    cores=[[1, 32], [1, 32]],
                    train_steps=[5000, 5000],
                    batch_nums=[64, 64],
                    learning_rates=[0.001, 0.001],
                    use_thresholds=[False, False],
                    thresholds=[5, 20],
                    retrain_time_limits=[4, 2],
                    thread_pool_size=6)
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
                           threshold_err=1,
                           threshold_err_cdf=10,
                           threshold_err_max_key=10,
                           is_retrain=True,
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
    logging.info("Error bound: %s" % index.model_err())
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
