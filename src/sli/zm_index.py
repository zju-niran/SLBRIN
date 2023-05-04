import logging
import math
import multiprocessing
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.mlp import MLP
from src.mlp_simple import MLPSimple
from src.spatial_index import SpatialIndex
from src.utils.common_utils import Region, biased_search_duplicate, normalize_input_minmax, \
    denormalize_output_minmax, binary_search_less_max, binary_search_duplicate, normalize_output, normalize_input, \
    relu, denormalize_outputs_minmax
from src.utils.geohash_utils import Geohash

PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class ZMIndex(SpatialIndex):
    """
    Z曲线学习型索引（Z-order model，ZM）
    Implement from Learned index for spatial queries
    """

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
        # for train
        self.weight = None
        self.cores = None
        self.train_step = None
        self.batch_num = None
        self.learning_rate = None
        # for compute
        self.io_cost = 0

    def build(self, data_list, is_sorted, data_precision, region, is_new, is_simple, weight,
              stages, cores, train_steps, batch_nums, learning_rates, use_thresholds, thresholds, retrain_time_limits,
              thread_pool_size):
        """
        build index
        1. ordering x/y point by geohash
        2. create rmi to train geohash->key data
        """
        self.weight = weight
        self.cores = cores[-1]
        self.train_step = train_steps[-1]
        self.batch_num = batch_nums[-1]
        self.learning_rate = learning_rates[-1]
        model_hdf_dir = os.path.join(self.model_path, "hdf/")
        if os.path.exists(model_hdf_dir) is False:
            os.makedirs(model_hdf_dir)
        model_png_dir = os.path.join(self.model_path, "../proposed_sli/png/")
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
            data_list = [(data[0], data[1], self.geohash.encode(data[0], data[1]), data[2], data[3])
                         for data in data_list]
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
                        pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple,
                                                    weight, core, train_step, batch_num, learning_rate,
                                                    use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(None, model, None) for model in mp_list]
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
                    pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple,
                                                weight, core, train_step, batch_num, learning_rate,
                                                use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(train_input[j], mp_list[j], Array()) for j in range(task_size)]
            self.rmi[i] = nodes
            # clear the data already used
            train_inputs[i] = None
            train_labels[i] = None

    def get_leaf_node(self, key):
        """
        get the leaf node which contains the key
        :param key: float
        :return: the key of leaf node
        """
        node_key = 0
        for i in range(0, self.non_leaf_stage_len):
            node_key = int(self.rmi[i][node_key].model.predict(key))
        return node_key

    def insert_single(self, point):
        """
        1. compute geohash from x/y of point
        2. encode p to geohash and create index entry(x, y, geohash, t, pointer)
        3. predict the leaf_node by rmi
        4. insert ie into delta index
        """
        # 1. compute geohash from x/y of point
        gh = self.geohash.encode(point[0], point[1])
        # 2. encode p to geohash and create index entry(x, y, geohash, t, pointer)
        point = (point[0], point[1], gh, point[2], point[3])
        # 3. predict the leaf_node by rmi
        node_key = self.get_leaf_node(gh)
        # 4. insert ie into delta index
        delta_index = self.rmi[-1][node_key].delta_index
        delta_index.insert(binary_search_less_max(delta_index.index, 2, gh, 0, delta_index.max_key) + 1, point)
        # IO1: search key
        self.io_cost += math.ceil((delta_index.max_key + 1) / ITEMS_PER_PAGE)

    def insert(self, points):
        points = points.tolist()
        for point in points:
            self.insert_single(point)

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
        node_key = self.get_leaf_node(key)
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
        node_key = self.get_leaf_node(key)
        leaf_model = self.rmi[-1][node_key].model
        if leaf_model is None:
            return 1
        return leaf_model.get_weight(key)

    def point_query_single(self, point):
        """
        1. compute geohash from x/y of point
        2. predict by geohash and create key scope [pre - min_err, pre + max_err]
        3. binary search in scope
        4. filter in delta index
        """
        # 1. compute geohash from x/y of point
        gh = self.geohash.encode(point[0], point[1])
        # 2. predict by geohash and create key scope [pre - min_err, pre + max_err]
        leaf_node, _, pre, min_err, max_err = self.predict(gh)
        l_bound = max(pre - max_err, leaf_node.model.output_min)
        r_bound = min(pre - min_err, leaf_node.model.output_max)
        # 3. binary search in scope
        result = [leaf_node.index[key][4] for key in
                  biased_search_duplicate(leaf_node.index, 2, gh, pre, l_bound, r_bound)]
        self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
        # 4. filter in delta index
        delta_index = leaf_node.delta_index
        if delta_index.max_key >= 0:
            result.extend([delta_index.index[key][4]
                           for key in
                           binary_search_duplicate(delta_index.index, 2, gh, 0, delta_index.max_key)])
            self.io_cost += math.ceil((delta_index.max_key + 1) / ITEMS_PER_PAGE)
        return result

    def range_query_single(self, window):
        """
        1. compute geohash from window_left and window_right
        2. find left_key by point query
        3. find right_key by point query
        4. filter all the points of scope[left_key, right_key] by range(x1/y1/x2/y2).contain(point)
        5. filter in delta index
        """
        # 1. compute z of window_left and window_right
        gh1 = self.geohash.encode(window[2], window[0])
        gh2 = self.geohash.encode(window[3], window[1])
        # 2. find left_key by point query
        # if point not found, left_key = pre - min_err
        leaf_node1, leaf_key1, pre1, min_err1, max_err1 = self.predict(gh1)
        l_bound1 = max(pre1 - max_err1, 0)
        r_bound1 = min(pre1 - min_err1, leaf_node1.model.output_max)
        left_key = biased_search_duplicate(leaf_node1.index, 2, gh1, pre1, l_bound1, r_bound1)
        left_key = l_bound1 if len(left_key) == 0 else min(left_key)
        # 3. find right_key by point query
        # if point not found, right_key = pre - max_err
        leaf_node2, leaf_key2, pre2, min_err2, max_err2 = self.predict(gh2)
        l_bound2 = max(pre2 - max_err2, 0)
        r_bound2 = min(pre2 - min_err2, leaf_node2.model.output_max)
        right_key = biased_search_duplicate(leaf_node2.index, 2, gh2, pre2, l_bound2, r_bound2)
        right_key = r_bound2 if len(right_key) == 0 else max(right_key)
        # 4. filter all the point of scope[key1, key2] by range(x1/y1/x2/y2).contain(point)
        # 5. filter in delta index
        io_index_len = 0
        io_delta_index_len = 0
        if leaf_key1 == leaf_key2:
            # filter index
            result = [ie[4] for ie in leaf_node1.index[left_key:right_key + 1]
                      if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
            io_index_len += r_bound2 - l_bound1
            # filter delta index
            delta_index = leaf_node1.delta_index
            if delta_index.max_key >= 0:
                left_key = binary_search_less_max(delta_index.index, 2, gh1, 0, delta_index.max_key)
                right_key = binary_search_less_max(delta_index.index, 2, gh2, left_key, delta_index.max_key)
                result.extend([ie[4]
                               for ie in delta_index.index[left_key:right_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_delta_index_len += delta_index.max_key + 1
        else:
            # filter index
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
            delta_index = leaf_node1.delta_index
            if delta_index.max_key >= 0:
                io_index_len += delta_index.max_key + 1
                left_key = binary_search_less_max(delta_index.index, 2, gh1, 0, delta_index.max_key)
                result.extend([ie[4]
                               for ie in delta_index.index[left_key:]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
            if leaf_key2 - leaf_key1 > 1:
                result.extend([ie[4]
                               for leaf_key in range(leaf_key1 + 1, leaf_key2)
                               for ie in self.rmi[-1][leaf_key].delta_index.index
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                for leaf_key in range(leaf_key1 + 1, leaf_key2):
                    io_index_len += len(self.rmi[-1][leaf_key].delta_index.index)
            delta_index = leaf_node2.delta_index
            if delta_index.max_key >= 0:
                io_index_len += delta_index.max_key + 1
                right_key = binary_search_less_max(delta_index.index, 2, gh1, 0, delta_index.max_key)
                result.extend([ie[4]
                               for ie in delta_index.index[:right_key + 1]
                               if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
        self.io_cost += math.ceil(io_index_len / ITEMS_PER_PAGE) + math.ceil(io_delta_index_len / ITEMS_PER_PAGE)
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
        k = int(k)
        w = self.get_weight(self.geohash.encode(x, y))
        if w > 0:
            window_ratio = (k / self.max_key) ** 0.5 / w
        else:
            window_ratio = (k / self.max_key) ** 0.5
        window_radius = window_ratio * self.geohash.region_width / 2
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
            left_key = biased_search_duplicate(leaf_node1.index, 2, gh1, pre1, l_bound1, r_bound1)
            left_key = l_bound1 if len(left_key) == 0 else min(left_key)
            leaf_node2, leaf_key2, pre2, min_err2, max_err2 = self.predict(gh2)
            l_bound2 = max(pre2 - max_err2, 0)
            r_bound2 = min(pre2 - min_err2, leaf_node2.model.output_max)
            right_key = biased_search_duplicate(leaf_node2.index, 2, gh2, pre2, l_bound2, r_bound2)
            right_key = r_bound2 + 1 if len(right_key) == 0 else max(right_key) + 1
            io_index_len = 0
            io_delta_index_len = 0
            if leaf_key1 == leaf_key2:
                tp_list = [ie for ie in leaf_node1.index[left_key:right_key]
                           if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
                io_index_len += right_key - left_key
                delta_index = leaf_node1.delta_index
                if delta_index.max_key >= 0:
                    delta_left_key = binary_search_less_max(delta_index.index, 2, gh1, 0, delta_index.max_key)
                    delta_right_key = binary_search_less_max(delta_index.index, 2, gh2, delta_left_key,
                                                             delta_index.max_key)
                    tp_list.extend([ie for ie in delta_index.index[delta_left_key:delta_right_key]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                    io_delta_index_len += delta_right_key - delta_left_key
            else:
                tp_list = [ie for ie in leaf_node1.index[left_key:]
                           if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]]
                io_index_len += len(leaf_node1.index) - left_key
                delta_index = leaf_node1.delta_index
                if delta_index.max_key >= 0:
                    delta_left_key = binary_search_less_max(delta_index.index, 2, gh1, 0, delta_index.max_key)
                    tp_list.extend([ie for ie in delta_index.index[delta_left_key:]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                    io_delta_index_len += delta_index.max_key - delta_left_key + 1
                tp_list.extend([ie for ie in leaf_node2.index[:right_key]
                                if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_index_len += len(leaf_node1.index) + right_key
                delta_index = leaf_node2.delta_index
                if delta_index.max_key >= 0:
                    delta_right_key = binary_search_less_max(delta_index.index, 2, gh2, 0, delta_index.max_key) + 1
                    tp_list.extend([ie for ie in delta_index.index[:delta_right_key]
                                    if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                    io_delta_index_len += delta_right_key
                tp_list.extend([ie for leaf_key in range(leaf_key1 + 1, leaf_key2)
                                for ie in self.rmi[-1][leaf_key].index
                                if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_index_len += sum([len(self.rmi[-1][leaf_key].index) for leaf_key in range(leaf_key1 + 1, leaf_key2)])
                tp_list.extend([ie for leaf_key in range(leaf_key1 + 1, leaf_key2)
                                for ie in self.rmi[-1][leaf_key].delta_index.index
                                if window[0] <= ie[1] <= window[1] and window[2] <= ie[0] <= window[3]])
                io_delta_index_len += sum(
                    [len(self.rmi[-1][leaf_key].delta_index.index) for leaf_key in range(leaf_key1 + 1, leaf_key2)])
            # 3. if target points is not enough, set window = 2 * window
            if len(tp_list) < k:
                window_radius *= 2
            else:
                # 4. elif target points is enough, but some target points is in the corner, set window = dst
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
        meta = np.array((self.geohash.data_precision,
                         self.geohash.region.bottom, self.geohash.region.up,
                         self.geohash.region.left, self.geohash.region.right,
                         self.weight, self.train_step, self.batch_num, self.learning_rate),
                        dtype=[("0", 'i4'),
                               ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8'),
                               ("5", 'f4'), ("6", 'i2'), ("7", 'i2'), ("8", 'f4')])
        np.save(os.path.join(self.model_path, 'meta.npy'), meta)
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
            delta_indexes.extend(node.delta_index.index)
            delta_index_lens.append(node.delta_index.size)
            delta_index_lens.append(node.delta_index.max_key)
        np.save(os.path.join(self.model_path, 'indexes.npy'),
                np.array(indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'index_lens.npy'), index_lens)
        np.save(os.path.join(self.model_path, 'delta_indexes.npy'),
                np.array(delta_indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'delta_index_lens.npy'), delta_index_lens)

    def load(self):
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
                self.rmi.append([Node(None, model, None) for model in models[model_cur:model_cur + self.stages[i]]])
                model_cur += self.stages[i]
            else:
                index_cur = 0
                delta_index_cur = 0
                delta_index_len_cur = 0
                leaf_nodes = []
                for j in range(self.stages[i]):
                    size = delta_index_lens[delta_index_len_cur]
                    max_key = delta_index_lens[delta_index_len_cur + 1]
                    index = delta_indexes[delta_index_cur:delta_index_cur + size]
                    leaf_nodes.append(Node(indexes[index_cur:index_cur + index_lens[j]],
                                           models[model_cur],
                                           Array(size, max_key, index)))
                    model_cur += 1
                    index_cur += index_lens[j]
                    delta_index_cur += size
                    delta_index_len_cur += 2
                self.rmi.append(leaf_nodes)

    def size(self):
        """
        structure_size = meta.npy + stages.npy + cores.npy + models.npy
        ie_size = index + delta_index
        """
        index_len = 0
        for leaf_node in self.rmi[-1]:
            index_len += len(leaf_node.index) + leaf_node.delta_index.size
        return os.path.getsize(os.path.join(self.model_path, "meta.npy")) - 128 - 64 * 2 + \
               os.path.getsize(os.path.join(self.model_path, "stages.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "cores.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "models.npy")) - 128, \
               index_len * ITEM_SIZE

    def model_err(self):
        return sum([(node.model.max_err - node.model.min_err) for node in self.rmi[-1] if node.model]) / self.stages[-1]

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
        if stage2_model_num + 1 < MODELS_PER_PAGE:
            model_io_list = [1] * stage2_model_num
        else:
            model_io_list = [1] * (MODELS_PER_PAGE - 1)
            model_io_list.extend([2] * (stage2_model_num + 1 - MODELS_PER_PAGE))
        # io when load data
        data_io_list = [math.ceil((node.model.max_err - node.model.min_err) / ITEMS_PER_PAGE) for node in self.rmi[-1]
                        if
                        node.model]
        # compute avg io: data io + node io
        data_num_list = [node.model.output_max - node.model.output_min + 1 for node in self.rmi[-1] if node.model]
        data_node_io_list = [(model_io_list[i] + data_io_list[i]) * data_num_list[i] for i in
                             range(stage2_model_num)]
        data_io = sum(data_node_io_list) / sum(data_num_list)
        # io when load update data
        update_data_num = sum([len(node.delta_index.index[:node.delta_index.max_key + 1]) for node in self.rmi[-1]])
        update_data_io_list = [math.ceil((node.delta_index.max_key + 1) / ITEMS_PER_PAGE) *
                               (node.delta_index.max_key + 1) for node in self.rmi[-1]]
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
                               None, None, None, None, None, None, None, None, None, None, None)
                tmp_index.clean_not_best_model_file()

    # def plot_model(self, stage_id, node_id):
    #     """
    #     plot the model
    #     """
    #     model_key = "%s_%s" % (stage_id, node_id)
    #     tmp_index = NN(self.model_path, model_key,
    #                    self.rmi[stage_id][node_id].index, None, None, None, None, None, None, None, None, None, None)
    #     tmp_index.plot()


def build_nn(model_path, curr_stage, current_stage_step, inputs, labels, is_new, is_simple,
             weight, core, train_step, batch_num, learning_rate,
             use_threshold, threshold, retrain_time_limit, mp_list=None):
    # In high stage, the data is too large to overflow in cpu/gpu, so adapt the batch_size normally by inputs
    batch_size = 2 ** math.ceil(math.log(len(inputs) / batch_num, 2))
    if batch_size < 1:
        batch_size = 1
    if is_simple:
        tmp_index = NNSimple(inputs, labels, weight, core, train_step, batch_size, learning_rate)
        tmp_index.build_simple()
    else:
        model_key = "%s_%s" % (curr_stage, current_stage_step)
        tmp_index = NN(model_path, model_key, inputs, labels, is_new,
                       weight, core, train_step, batch_size, learning_rate,
                       use_threshold, threshold, retrain_time_limit)
        tmp_index.build()
    abstract_index = AbstractNN(tmp_index.get_matrices(), len(core) - 1,
                                int(tmp_index.train_x_min), int(tmp_index.train_x_max),
                                int(tmp_index.train_y_min), int(tmp_index.train_y_max),
                                math.floor(tmp_index.min_err), math.ceil(tmp_index.max_err))
    mp_list[current_stage_step] = abstract_index


class Node:
    def __init__(self, index, model, delta_index, delta_model=None):
        self.index = index
        self.model = model
        self.delta_index = delta_index
        self.delta_model = delta_model


class NN(MLP):
    def __init__(self, model_path, model_key, train_x, train_y, is_new, weight, core, train_step, batch_size,
                 learning_rate, use_threshold, threshold, retrain_time_limit):
        self.name = "ZM Index NN"
        # 当只有一个输入输出时，整数的key作为y_true会导致loss中y_true-y_pred出现类型错误：
        # TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
        train_x, train_x_min, train_x_max = normalize_input(np.array(train_x).astype("float"))
        train_y, train_y_min, train_y_max = normalize_output(np.array(train_y).astype("float"))
        super().__init__(model_path, model_key, train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         is_new, weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                         retrain_time_limit)


class NNSimple(MLPSimple):
    def __init__(self, train_x, train_y, weight, core, train_step, batch_size, learning_rate):
        self.name = "ZM Index NN"
        # 当只有一个输入输出时，整数的key作为y_true会导致loss中y_true-y_pred出现类型错误：
        # TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
        train_x, train_x_min, train_x_max = normalize_input(np.array(train_x).astype("float"))
        train_y, train_y_min, train_y_max = normalize_output(np.array(train_y).astype("float"))
        super().__init__(train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         weight, core, train_step, batch_size, learning_rate)


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
        return (np.dot(y2, self.matrices[-2]) - np.dot(y1, self.matrices[-2]))[0, 0] / delta

    def update_error_range(self, xs):
        xs_len = len(xs)
        self.output_max = xs_len - 1
        if xs_len:
            # 数据量太多，predict很慢，因此用均匀采样得到100个点来计算误差
            if xs_len > 100:
                step_size = xs_len // 100
                sample_keys = [i for i in range(0, step_size * 100, step_size)]
                xs = np.array([xs[i] for i in sample_keys])
                ys = np.array(sample_keys)
            else:
                xs = np.array(xs)
                ys = np.arange(xs_len)
            pres = normalize_input_minmax(np.expand_dims(xs, -1), self.input_min, self.input_max)
            for i in range(self.hl_nums):
                pres = relu(np.dot(pres, self.matrices[i * 2]) + self.matrices[i * 2 + 1])
            pres = np.dot(pres, self.matrices[-2]) + self.matrices[-1]
            errs = denormalize_outputs_minmax(pres.flatten(), ys.min(), ys.max()) - ys
            self.min_err = math.floor(errs.min())
            self.max_err = math.ceil(errs.max())
        else:
            self.min_err = 0
            self.max_err = 0


class Array:
    """
    模拟python数组：
    1. 初始化：1个Page
    2. 扩容：每次扩容增大原来的1/8
    3. 插入：检查是否需要扩容，右移插入点后的所有数据，返回移动的数据数量
    """

    def __init__(self, size=ITEMS_PER_PAGE, max_key=-1, index=None):
        self.size = size
        self.max_key = max_key
        self.index = [(0, 0, 0, 0, 0) for i in range(size)] if index is None else index

    def expand(self, size=None):
        if size is None:
            size = max(int(self.size / 8), 1)
        self.index.extend([(0, 0, 0, 0, 0) for i in range(size)])
        self.size += size

    def insert(self, key, value):
        self.max_key += 1
        if self.max_key == self.size:
            self.expand()
        for i in range(self.max_key, key, -1):
            self.index[i] = self.index[i - 1]
        self.index[key] = value
        return self.max_key - key + 1


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/zm_index_10w/"
    data_distribution = Distribution.NYCT_10W_SORTED
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
