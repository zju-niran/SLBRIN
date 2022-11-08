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
from src.mlp import MLP
from src.mlp_simple import MLPSimple
from src.spatial_index.common_utils import Region, biased_search, normalize_input_minmax, denormalize_output_minmax, \
    binary_search_less_max, binary_search, relu, normalize_output, normalize_input
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query

# 预设pagesize=4096, read_ahead_pages=256, size(model)=2000, size(pointer)=4, size(x/y/geohash)=8
RA_PAGES = 256
PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_RA = RA_PAGES * int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_RA = RA_PAGES * int(PAGE_SIZE / ITEM_SIZE)


class ZMIndex(SpatialIndex):
    def __init__(self, model_path=None):
        super(ZMIndex, self).__init__("ZM Index")
        self.geohash = None
        self.stages = None
        self.non_leaf_stage_len = 0
        self.max_key = 0
        self.rmi = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # 训练所需：
        self.is_gpu = None
        self.weight = None
        self.cores = None
        self.train_step = None
        self.batch_num = None
        self.learning_rate = None
        # 统计所需：
        self.io_cost = 0

    def build(self, data_list, is_sorted, data_precision, region, is_new, is_simple, is_gpu, weight,
              stages, cores, train_steps, batch_nums, learning_rates, use_thresholds, thresholds, retrain_time_limits,
              thread_pool_size):
        """
        build index
        1. ordering x/y point by geohash
        2. create rmi to train geohash->key data
        """
        self.is_gpu = is_gpu
        self.weight = weight
        self.cores = cores[-1]
        self.train_step = train_steps[-1]
        self.batch_num = batch_nums[-1]
        self.learning_rate = learning_rates[-1]
        model_hdf_dir = os.path.join(self.model_path, "hdf/")
        if os.path.exists(model_hdf_dir) is False:
            os.makedirs(model_hdf_dir)
        model_png_dir = os.path.join(self.model_path, "png/")
        if os.path.exists(model_png_dir) is False:
            os.makedirs(model_png_dir)
        self.geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        self.stages = stages
        stage_len = len(stages)
        self.non_leaf_stage_len = stage_len - 1
        train_inputs = [[[] for j in range(stages[i])] for i in range(stage_len)]
        train_labels = [[[] for j in range(stages[i])] for i in range(stage_len)]
        self.rmi = [None for i in range(stage_len)]
        # 1. ordering x/y point by geohash
        data_len = len(data_list)
        self.max_key = data_len
        if not is_sorted:
            data_list = [(data_list[i][0], data_list[i][1], self.geohash.encode(data_list[i][0], data_list[i][1]), i)
                         for i in range(0, data_len)]
            data_list = sorted(data_list, key=lambda x: x[2])
        else:
            data_list = data_list.tolist()
        train_inputs[0][0] = data_list
        train_labels[0][0] = list(range(0, data_len))
        # 2. create rmi to train geohash->key data
        for i in range(stage_len):
            core = cores[i]
            train_step = train_steps[i]
            batch_num = batch_nums[i]
            learning_rate = learning_rates[i]
            use_threshold = use_thresholds[i]
            threshold = thresholds[i]
            retrain_time_limit = retrain_time_limits[i]
            pool = multiprocessing.Pool(processes=thread_pool_size)
            task_size = stages[i]
            mp_list = multiprocessing.Manager().list([None] * task_size)
            train_input = train_inputs[i]
            train_label = train_labels[i]
            # 2.1 create non-leaf node
            if i < self.non_leaf_stage_len:
                for j in range(task_size):
                    if train_label[j] is None:
                        continue
                    else:
                        # build inputs
                        inputs = [data[2] for data in train_input[j]]
                        # build labels
                        divisor = stages[i + 1] * 1.0 / data_len
                        labels = [int(k * divisor) for k in train_label[j]]
                        # train model
                        pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple, is_gpu,
                                                    weight, core, train_step, batch_num, learning_rate,
                                                    use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(model, None, None) for model in mp_list]
                for j in range(task_size):
                    node = nodes[j]
                    if node is None:
                        continue
                    else:
                        # predict and build inputs and labels for next stage
                        for ind in range(len(train_input[j])):
                            # pick model in next stage with output of this model
                            pre = int(node.model.predict(train_input[j][ind][2]))
                            train_inputs[i + 1][pre].append(train_input[j][ind])
                            train_labels[i + 1][pre].append(train_label[j][ind])
            # 2.2 create leaf node
            else:
                for j in range(task_size):
                    inputs = [data[2] for data in train_input[j]]
                    labels = list(range(0, len(inputs)))
                    if not labels:
                        continue
                    pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple, is_gpu,
                                                weight, core, train_step, batch_num, learning_rate,
                                                use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(mp_list[j], train_input[j], []) for j in range(task_size)]
            self.rmi[i] = nodes
            # clear the data already used
            train_inputs[i] = None
            train_labels[i] = None

    def insert_single(self, point):
        """
        1. compute geohash from x/y of point
        2. encode p to geohash and create index entry(x, y, geohash, pointer)
        3. predict the leaf_node by rmi
        4. insert ie into update index
        """
        # 1. compute geohash from x/y of point
        gh = self.geohash.encode(point[0], point[1])
        # 2. encode p to geohash and create index entry(x, y, geohash, pointer)
        point = (point[0], point[1], gh, point[2])
        # 3. predict the leaf_node by rmi
        node_key = 0
        for i in range(0, self.non_leaf_stage_len):
            node_key = int(self.rmi[i][node_key].model.predict(gh))
        # 4. insert ie into update index
        leaf_node = self.rmi[-1][node_key]
        leaf_node.delta_index.insert(
            binary_search_less_max(leaf_node.index, 2, gh, 0, len(leaf_node.index) - 1) + 1, point)

    def insert(self, points):
        points = points.tolist()
        for point in points:
            self.insert_single(point)

    def update(self):
        leaf_nodes = self.rmi[-1]
        for j in range(0, self.stages[-1]):
            leaf_node = leaf_nodes[j]
            if leaf_node.delta_index:
                # 1. merge delta index into index
                if leaf_node.index:
                    leaf_node.index.extend(leaf_node.delta_index)
                    leaf_node.index.sort(key=lambda x: x[2])  # 优化：有序数组合并->sorted:2.5->1
                else:
                    leaf_node.index = leaf_node.delta_index
                # 2. retrain model
                inputs = [data[2] for data in leaf_node.index]
                inputs_num = len(inputs)
                labels = list(range(0, inputs_num))
                batch_size = 2 ** math.ceil(math.log(inputs_num / self.batch_num, 2))
                if batch_size < 1:
                    batch_size = 1
                model_key = "retrain_%s" % j
                tmp_index = NN(self.model_path, model_key, inputs, labels, True, self.is_gpu, self.weight,
                               self.cores, self.train_step, batch_size, self.learning_rate, False, None, None)
                tmp_index.train_simple(None)  # update
                # tmp_index.train_simple(leaf_node.model.matrices if leaf_node.model else None) # retrain with
                leaf_node.model = AbstractNN(tmp_index.get_matrices(), len(self.cores) - 1,
                                             int(tmp_index.train_x_min), int(tmp_index.train_x_max),
                                             0, inputs_num - 1,  # update the range of output
                                             math.ceil(tmp_index.min_err), math.ceil(tmp_index.max_err))
                self.logging.info("retrain leaf model %s" % model_key)

    def predict(self, key):
        """
        predict key from key
        1. predict the leaf_node by rmi
        2. return the less max key when leaf model is None
        3. predict the key by leaf_node
        :param key: float
        :return: leaf node, the key predicted by rmi, left and right err bounds
        """
        # 1. predict the leaf_node by rmi
        node_key = 0
        for i in range(0, self.non_leaf_stage_len):
            node_key = int(self.rmi[i][node_key].model.predict(key))
        # 2. return the less max key when leaf model is None
        leaf_node = self.rmi[-1][node_key]
        if leaf_node.model is None:
            while self.rmi[-1][node_key].model is None:
                node_key -= 1
                if node_key <= 0:
                    break
            return self.rmi[-1][node_key], node_key, self.rmi[-1][node_key].model.output_max, 0, 0
        # 3. predict the key by leaf_node
        pre = int(leaf_node.model.predict(key))
        return leaf_node, node_key, pre, leaf_node.model.min_err, leaf_node.model.max_err

    def get_weight(self, key):
        """
        calculate weight from key
        uniform分布的斜率理论上为1，密集分布则大于1，稀疏分布则小于1
        """
        node_key = 0
        for i in range(0, self.non_leaf_stage_len):
            node_key = int(self.rmi[i][node_key].model.predict(key))
        leaf_model = self.rmi[-1][node_key].model
        if leaf_model is None:
            return 1
        return leaf_model.get_weight(key)

    def point_query_single(self, point):
        """
        1. compute geohash from x/y of point
        2. predict by geohash and create key scope [pre - min_err, pre + max_err]
        3. binary search in scope
        4. filter in update index
        """
        # 1. compute geohash from x/y of point
        gh = self.geohash.encode(point[0], point[1])
        # 2. predict by geohash and create key scope [pre - min_err, pre + max_err]
        leaf_node, _, pre, min_err, max_err = self.predict(gh)
        l_bound = max(pre - max_err, 0)
        r_bound = min(pre - min_err, leaf_node.model.output_max)
        # 3. binary search in scope
        result = [leaf_node.index[key][3] for key in biased_search(leaf_node.index, 2, gh, pre, l_bound, r_bound)]
        self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_RA)
        # 4. filter in update index
        if leaf_node.delta_index:
            delta_index_len = len(leaf_node.delta_index)
            result.extend([leaf_node.delta_index[key][3]
                           for key in binary_search(leaf_node.delta_index, 2, gh, 0, len(leaf_node.delta_index) - 1)])
            self.io_cost += delta_index_len // ITEMS_PER_RA + 1
        return result

    def range_query_single(self, window):
        """
        1. compute geohash from window_left and window_right
        2. find left_key by point query
        3. find right_key by point query
        4. filter all the points of scope[left_key, right_key] by range(x1/y1/x2/y2).contain(point)
        5. filter in update index
        """
        # 1. compute z of window_left and window_right
        gh1 = self.geohash.encode(window[2], window[0])
        gh2 = self.geohash.encode(window[3], window[1])
        # 2. find left_key by point query
        # if point not found, left_key = pre - min_err
        leaf_node1, leaf_key1, pre1, min_err1, max_err1 = self.predict(gh1)
        l_bound1 = max(pre1 - max_err1, 0)
        r_bound1 = min(pre1 - min_err1, leaf_node1.model.output_max)
        left_key = biased_search(leaf_node1.index, 2, gh1, pre1, l_bound1, r_bound1)
        left_key = l_bound1 if len(left_key) == 0 else min(left_key)
        # 3. find right_key by point query
        # if point not found, right_key = pre - max_err
        leaf_node2, leaf_key2, pre2, min_err2, max_err2 = self.predict(gh2)
        l_bound2 = max(pre2 - max_err2, 0)
        r_bound2 = min(pre2 - min_err2, leaf_node2.model.output_max)
        right_key = biased_search(leaf_node2.index, 2, gh2, pre2, l_bound2, r_bound2)
        right_key = r_bound2 if len(right_key) == 0 else max(right_key)
        # 4. filter all the point of scope[key1, key2] by range(x1/y1/x2/y2).contain(point)
        # 5. filter in update index
        if leaf_key1 == leaf_key2:
            # filter index
            result = [ie[3] for ie in leaf_node1.index[left_key:right_key + 1]
                      if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
            self.io_cost += math.ceil((r_bound2 - l_bound1) / ITEMS_PER_RA)
            # filter delta index
            delta_index = leaf_node1.delta_index
            if delta_index:
                delta_index_len = len(delta_index)
                left_key = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                right_key = binary_search_less_max(delta_index, 2, gh2, left_key, delta_index_len - 1)
                result.extend([ie[3]
                               for ie in delta_index[left_key:right_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                self.io_cost += math.ceil(delta_index_len / ITEMS_PER_RA)
        else:
            # filter index
            io_index_len = len(leaf_node1.index) + r_bound2 - l_bound1
            result = [ie[3]
                      for ie in leaf_node1.index[left_key:]
                      if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
            if leaf_key2 - leaf_key1 > 1:
                result.extend([ie[3]
                               for leaf_key in range(leaf_key1 + 1, leaf_key2)
                               for ie in self.rmi[-1][leaf_key].index
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                for leaf_key in range(leaf_key1 + 1, leaf_key2):
                    io_index_len += len(self.rmi[-1][leaf_key].index)
            result.extend([ie[3]
                           for ie in leaf_node2.index[:right_key + 1]
                           if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
            self.io_cost += math.ceil(io_index_len / ITEMS_PER_RA)
            # filter delta index
            delta_index = leaf_node1.delta_index
            io_index_len = 0
            if delta_index:
                delta_index_len = len(delta_index)
                io_index_len += delta_index_len
                left_key = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                result.extend([ie[3]
                               for ie in delta_index[left_key:]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
            if leaf_key2 - leaf_key1 > 1:
                result.extend([ie[3]
                               for leaf_key in range(leaf_key1 + 1, leaf_key2)
                               for ie in self.rmi[-1][leaf_key].delta_index
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                for leaf_key in range(leaf_key1 + 1, leaf_key2):
                    io_index_len += len(self.rmi[-1][leaf_key].delta_index)
            delta_index = leaf_node2.delta_index
            if delta_index:
                delta_index_len = len(delta_index)
                io_index_len += delta_index_len
                key_right = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                right_key = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                result.extend([ie[3]
                               for ie in delta_index[:right_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
            self.io_cost += math.ceil(io_index_len / ITEMS_PER_RA)
        return result

    def knn_query_single(self, knn):
        """
        1. init window by weight on CDF(key)
        2. iter: query target points by range query
        3. if target points is not enough, set window = 2 * window
        4. elif target points is enough, but some target points is in the corner, set window = dst
        """
        # 1. init window by weight on CDF(key)
        x, y, k = knn
        w = self.get_weight(self.geohash.encode(x, y))
        if w > 0:
            window_ratio = (k / self.max_key) ** 0.5 / w
        else:
            window_ratio = (k / self.max_key) ** 0.5
        window_radius = window_ratio * self.geohash.region_width / 2
        tp_list = []
        old_window = None
        while True:
            window = [y - window_radius, y + window_radius, x - window_radius, x + window_radius]
            # limit window within region
            self.geohash.region.clip_region(window, self.geohash.data_precision)
            # 2. iter: query target points by range query
            gh1 = self.geohash.encode(window[2], window[0])
            gh2 = self.geohash.encode(window[3], window[1])
            leaf_node1, leaf_key1, pre1, min_err1, max_err1 = self.predict(gh1)
            l_bound1 = max(pre1 - max_err1, 0)
            r_bound1 = min(pre1 - min_err1, leaf_node1.model.output_max)
            left_key = biased_search(leaf_node1.index, 2, gh1, pre1, l_bound1, r_bound1)
            left_key = l_bound1 if len(left_key) == 0 else min(left_key)
            leaf_node2, leaf_key2, pre2, min_err2, max_err2 = self.predict(gh2)
            l_bound2 = max(pre2 - max_err2, 0)
            r_bound2 = min(pre2 - min_err2, leaf_node2.model.output_max)
            key_right = biased_search(leaf_node2.index, 2, gh2, pre2, l_bound2, r_bound2)
            key_right = r_bound2 if len(key_right) == 0 else max(key_right)
            if old_window:
                filter_lambda = lambda ie: window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3] and not (
                        old_window[0] <= ie[1] <= old_window[1] and old_window[2] <= ie[0] <= old_window[3])
            else:
                filter_lambda = lambda ie: window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]
            if leaf_key1 == leaf_key2:
                tmp_tp_list = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                               for ie in leaf_node1.index[key_left:key_right + 1] if filter_lambda(ie)]
                self.io_cost += math.ceil((r_bound2 - l_bound1) / ITEMS_PER_RA)
                delta_index = leaf_node1.delta_index
                if delta_index:
                    delta_index_len = len(delta_index)
                    key_left = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                    key_right = binary_search_less_max(delta_index, 2, gh2, key_left, delta_index_len - 1)
                    tmp_tp_list.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                                        for ie in delta_index[key_left:key_right + 1] if filter_lambda(ie)])
                    self.io_cost += math.ceil(delta_index_len / ITEMS_PER_RA)
            else:
                io_index_len = len(leaf_node1.index) + r_bound2 - l_bound1
                result = [[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                          for ie in leaf_node1.index[key_left:] if filter_lambda(ie)]
                if leaf_key2 - leaf_key1 > 1:
                    result.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                                   for leaf_key in range(leaf_key1 + 1, leaf_key2)
                                   for ie in self.rmi[-1][leaf_key].index if filter_lambda(ie)])
                    for leaf_key in range(leaf_key1 + 1, leaf_key2):
                        io_index_len += len(self.rmi[-1][leaf_key].index)
                result.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                               for ie in leaf_node2.index[:key_right + 1] if filter_lambda(ie)])
                self.io_cost += math.ceil(io_index_len / ITEMS_PER_RA)
                # filter delta index
                delta_index = leaf_node1.delta_index
                io_index_len = 0
                if delta_index:
                    delta_index_len = len(delta_index)
                    io_index_len += delta_index_len
                    key_left = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                    result.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                                   for ie in delta_index[key_left:] if filter_lambda(ie)])
                if leaf_key2 - leaf_key1 > 1:
                    result.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                                   for leaf_key in range(leaf_key1 + 1, leaf_key2)
                                   for ie in self.rmi[-1][leaf_key].delta_index if filter_lambda(ie)])
                    for leaf_key in range(leaf_key1 + 1, leaf_key2):
                        io_index_len += len(self.rmi[-1][leaf_key].delta_index)
                delta_index = leaf_node2.delta_index
                if delta_index:
                    delta_index_len = len(delta_index)
                    io_index_len += delta_index_len
                    key_right = binary_search_less_max(delta_index, 2, gh1, 0, delta_index_len - 1)
                    result.extend([[(ie[0] - x) ** 2 + (ie[1] - y) ** 2, ie[3]]
                                   for ie in delta_index[:key_right + 1] if filter_lambda(ie)])
                    self.io_cost += math.ceil(io_index_len / ITEMS_PER_RA)
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
        zmin_meta = np.array((self.geohash.data_precision,
                              self.geohash.region.bottom, self.geohash.region.up,
                              self.geohash.region.left, self.geohash.region.right,
                              self.is_gpu, self.weight, self.train_step, self.batch_num, self.learning_rate),
                             dtype=[("0", 'i4'),
                                    ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8'),
                                    ("5", 'i1'), ("6", 'f4'), ("7", 'i2'), ("8", 'i2'), ("9", 'f4')])
        np.save(os.path.join(self.model_path, 'meta.npy'), zmin_meta)
        np.save(os.path.join(self.model_path, 'stages.npy'), self.stages)
        np.save(os.path.join(self.model_path, 'cores.npy'), self.cores)
        models = []
        for stage in self.rmi:
            models.extend([node.model for node in stage])
        np.save(os.path.join(self.model_path, 'models.npy'), models)
        indexes = []
        index_lens = []
        delta_indexes = []
        delta_index_lens = []
        for node in self.rmi[-1]:
            indexes.extend(node.index)
            index_lens.append(len(node.index))
            delta_indexes.extend(node.delta_index)
            delta_index_lens.append(len(node.delta_index))
        np.save(os.path.join(self.model_path, 'indexes.npy'),
                np.array(indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4')]))
        np.save(os.path.join(self.model_path, 'index_lens.npy'), index_lens)
        np.save(os.path.join(self.model_path, 'delta_indexes.npy'),
                np.array(delta_indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4')]))
        np.save(os.path.join(self.model_path, 'delta_index_lens.npy'), delta_index_lens)

    def load(self):
        meta = np.load(os.path.join(self.model_path, 'meta.npy'), allow_pickle=True).item()
        region = Region(meta[1], meta[2], meta[3], meta[4])
        self.geohash = Geohash.init_by_precision(data_precision=meta[0], region=region)
        self.stages = np.load(os.path.join(self.model_path, 'stages.npy'), allow_pickle=True).tolist()
        self.non_leaf_stage_len = len(self.stages) - 1
        self.cores = np.load(os.path.join(self.model_path, 'cores.npy'), allow_pickle=True).tolist()
        self.is_gpu = bool(meta[5])
        self.weight = meta[6]
        self.train_step = meta[7]
        self.batch_num = meta[8]
        self.learning_rate = meta[9]
        models = np.load(os.path.join(self.model_path, 'models.npy'), allow_pickle=True)
        indexes = np.load(os.path.join(self.model_path, 'indexes.npy'), allow_pickle=True).tolist()
        index_lens = np.load(os.path.join(self.model_path, 'index_lens.npy'), allow_pickle=True).tolist()
        delta_indexes = np.load(os.path.join(self.model_path, 'delta_indexes.npy'), allow_pickle=True).tolist()
        delta_index_lens = np.load(os.path.join(self.model_path, 'delta_index_lens.npy'), allow_pickle=True).tolist()
        self.max_key = len(indexes)
        model_cur = 0
        self.rmi = []
        for i in range(len(self.stages)):
            if i < self.non_leaf_stage_len:
                self.rmi.append([Node(model, None, None) for model in models[model_cur:model_cur + self.stages[i]]])
                model_cur += self.stages[i]
            else:
                index_cur = 0
                delta_index_cur = 0
                leaf_nodes = []
                for j in range(self.stages[i]):
                    leaf_nodes.append(Node(models[model_cur],
                                           indexes[index_cur:index_cur + index_lens[j]],
                                           delta_indexes[delta_index_cur:delta_index_cur + delta_index_lens[j]]))
                    model_cur += 1
                    index_cur += index_lens[j]
                    delta_index_cur += delta_index_lens[j]
                self.rmi.append(leaf_nodes)
        self.io_cost = math.ceil(self.size()[0] / ITEMS_PER_RA)

    def size(self):
        """
        structure_size = meta.npy + stages.npy + cores.npy + models.npy
        ie_size = index_lens.npy + indexes.npy + delta_index_lens.npy + delta_indexes.npy
        """
        return os.path.getsize(os.path.join(self.model_path, "meta.npy")) - 128 - 64 * 2 + \
               os.path.getsize(os.path.join(self.model_path, "stages.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "cores.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "models.npy")) - 128, \
               os.path.getsize(os.path.join(self.model_path, "index_lens.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "indexes.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "delta_index_lens.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "delta_indexes.npy")) - 128

    def avg_io_cost(self):
        """
        假设查询条件和数据分布一致，io=获取meta的io+获取stage1 node的io+获取stage2 node的io+获取data的io+获取update data的io
        一次read_ahead可以拿512个node，因此前面511个stage2 node的io是1，后面统一为2
        data io由model误差范围决定，update data io由model update部分的数据量决定
        先计算单个node的node io和data io，然后乘以node的数据量，最后除以总数据量，来计算整体的平均io
        """
        stage2_model_num = len(
            [node.model for node in self.rmi[-1] if node.model]) if self.non_leaf_stage_len > 0 else 0
        # io when load node
        if stage2_model_num + 1 < MODELS_PER_RA:
            model_io_list = [1] * stage2_model_num
        else:
            model_io_list = [1] * (MODELS_PER_RA - 1)
            model_io_list.extend([2] * (stage2_model_num + 1 - MODELS_PER_RA))
        # io when load data
        data_io_list = [math.ceil((node.model.max_err - node.model.min_err) / ITEMS_PER_RA) for node in self.rmi[-1] if
                        node.model]
        # compute avg io: data io + node io
        data_num_list = [node.model.output_max - node.model.output_min + 1 for node in self.rmi[-1] if node.model]
        data_node_io_list = [(model_io_list[i] + data_io_list[i]) * data_num_list[i] for i in
                             range(stage2_model_num)]
        data_io = sum(data_node_io_list) / sum(data_num_list)
        # io when load update data
        update_data_num = sum([len(node.delta_index) for node in self.rmi[-1]])
        update_data_io_list = [math.ceil(len(node.delta_index) / ITEMS_PER_RA) * len(node.delta_index) for node in
                               self.rmi[-1]]
        update_data_io = sum(update_data_io_list) / update_data_num if update_data_num else 0
        return data_io + update_data_io

    def model_clear(self):
        """
        clear the models which are not the best
        """
        for i in range(0, self.non_leaf_stage_len + 1):
            for j in range(len(self.rmi[i])):
                model_key = "%s_%s" % (i, j)
                tmp_index = NN(self.model_path, model_key,
                               None, None, None, None, None, None, None, None, None, None, None, None)
                tmp_index.clean_not_best_model_file()


def build_nn(model_path, curr_stage, current_stage_step, inputs, labels, is_new, is_simple, is_gpu,
             weight, core, train_step, batch_num, learning_rate,
             use_threshold, threshold, retrain_time_limit, mp_list=None):
    # In high stage, the data is too large to overflow in cpu/gpu, so adapt the batch_size normally by inputs
    batch_size = 2 ** math.ceil(math.log(len(inputs) / batch_num, 2))
    if batch_size < 1:
        batch_size = 1
    if is_simple:
        tmp_index = NNSimple(inputs, labels, is_gpu, weight, core, train_step, batch_size, learning_rate)
    else:
        model_key = "%s_%s" % (curr_stage, current_stage_step)
        tmp_index = NN(model_path, model_key, inputs, labels, is_new, is_gpu,
                       weight, core, train_step, batch_size, learning_rate,
                       use_threshold, threshold, retrain_time_limit)
    tmp_index.train()
    abstract_index = AbstractNN(tmp_index.get_matrices(), len(core) - 1,
                                int(tmp_index.train_x_min), int(tmp_index.train_x_max),
                                int(tmp_index.train_y_min), int(tmp_index.train_y_max),
                                math.ceil(tmp_index.min_err), math.ceil(tmp_index.max_err))
    del tmp_index
    gc.collect(generation=0)
    mp_list[current_stage_step] = abstract_index


class Node:
    def __init__(self, model, index, delta_index):
        self.model = model
        self.index = index
        self.delta_index = delta_index


class NN(MLP):
    def __init__(self, model_path, model_key, train_x, train_y, is_new, is_gpu, weight, core, train_step, batch_size,
                 learning_rate, use_threshold, threshold, retrain_time_limit):
        self.name = "ZM Index NN"
        # 当只有一个输入输出时，整数的key作为y_true会导致loss中y_true-y_pred出现类型错误：
        # TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
        train_x, train_x_min, train_x_max = normalize_input(np.array(train_x).astype("float"))
        train_y, train_y_min, train_y_max = normalize_output(np.array(train_y).astype("float"))
        super().__init__(model_path, model_key, train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         is_new, is_gpu, weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                         retrain_time_limit)


class NNSimple(MLPSimple):
    def __init__(self, train_x, train_y, is_gpu, weight, core, train_step, batch_size, learning_rate):
        self.name = "ZM Index NN"
        # 当只有一个输入输出时，整数的key作为y_true会导致loss中y_true-y_pred出现类型错误：
        # TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
        train_x, train_x_min, train_x_max = normalize_input(np.array(train_x).astype("float"))
        train_y, train_y_min, train_y_max = normalize_output(np.array(train_y).astype("float"))
        super().__init__(train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         is_gpu, weight, core, train_step, batch_size, learning_rate)


class AbstractNN:
    def __init__(self, matrices, hl_nums, input_min, input_max, output_min, output_max, min_err, max_err):
        self.matrices = matrices
        self.hl_nums = hl_nums
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max
        self.min_err = min_err
        self.max_err = max_err

    # compared with *, np.dot is a little slower, but closer to nn.predict
    def predict(self, input_key):
        y = normalize_input_minmax(input_key, self.input_min, self.input_max)
        for i in range(self.hl_nums):
            y = relu(np.dot(y, self.matrices[i * 2]) + self.matrices[i * 2 + 1])
        y = np.dot(y, self.matrices[-2]) + self.matrices[-1]
        return denormalize_output_minmax(y[0, 0], self.output_min, self.output_max)

    def get_weight(self, input_key):
        """
        calculate weight
        """
        # delta当前选8位有效数字，是matrix的最高精度
        delta = 0.00000001
        y1 = normalize_input_minmax(input_key, self.input_min, self.input_max)
        y2 = y1 + delta
        for i in range(self.hl_nums):
            y1 = relu(y1 * self.matrices[i * 2] + self.matrices[i * 2 + 1])
            y2 = relu(y2 * self.matrices[i * 2] + self.matrices[i * 2 + 1])
        return np.dot((y2 - y1), self.matrices[-2])[0, 0] / delta


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/zm_index_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = ZMIndex(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        data_distribution = Distribution.NYCT_10W_SORTED
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    is_new=False,
                    is_simple=False,
                    is_gpu=True,
                    weight=1,
                    stages=[1, 100],
                    cores=[[1, 32], [1, 32]],
                    train_steps=[5000, 5000],
                    batch_nums=[64, 64],
                    learning_rates=[0.1, 0.1],
                    use_thresholds=[False, False],
                    thresholds=[5, 20],
                    retrain_time_limits=[4, 2],
                    thread_pool_size=6)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    logging.info("IO cost: %s" % index.get_io_cost())
    io_cost = index.io_cost
    logging.info("IO cost: %s" % io_cost)
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
    path = '../../data/query/range_query_nyct.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(range_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/knn_query_nyct.npy'
    knn_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(knn_query_list)))
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))
    start_time = time.time()
    index.update()
    end_time = time.time()
    logging.info("Update time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
