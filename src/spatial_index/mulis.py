import gc
import logging
import math
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/SBRIN')
from src.mlp import MLP
from src.mlp_simple import MLPSimple
from src.spatial_index.common_utils import Region, biased_search, normalize_input_minmax, denormalize_output_minmax, \
    binary_search_less_max, binary_search, relu
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex
from src.ts_model import TimeSeriesModel, build_cdf
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query

# 预设pagesize=4096, read_ahead_pages=256, size(model)=2000, size(pointer)=4, size(x/y/geohash)=8
RA_PAGES = 256
PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_RA = RA_PAGES * int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_RA = RA_PAGES * int(PAGE_SIZE / ITEM_SIZE)


class Mulis(SpatialIndex):
    def __init__(self, model_path=None):
        super(Mulis, self).__init__("MULIS")
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
        # 更新所需：
        self.cdf_width = None
        self.cdf_lag = None
        self.start_time = None
        self.cur_time_interval = None
        self.time_interval = None
        # 训练所需：
        self.is_gpu = None
        self.weight = None
        self.cores = None
        self.train_step = None
        self.batch_num = None
        self.learning_rate = None

    def build(self, data_list, is_sorted, data_precision, region, is_new, is_simple, is_gpu, weight, stages, cores,
              train_steps, batch_nums, learning_rates, use_thresholds, thresholds, retrain_time_limits,
              thread_pool_size,
              time_interval, start_time, end_time, cdf_width, cdf_lag):
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
        self.cdf_width = cdf_width
        self.cdf_lag = cdf_lag
        self.start_time = start_time
        self.cur_time_interval = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
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
                        pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple, is_gpu,
                                                    weight, core, train_step, batch_num, learning_rate,
                                                    use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(None, model, None, None) for model in mp_list]
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
                # 2.2.1 create index and model
                # get the key bound of leaf nodes
                key_left_bounds = self.get_leaf_bound()
                for j in range(task_size):
                    inputs = [data[2] for data in train_input[j]]
                    # add the key bound into inputs
                    left_bound = key_left_bounds[j]
                    right_bound = (key_left_bounds[j + 1] if j + 1 < task_size else 1 << self.geohash.sum_bits)
                    if inputs and not (left_bound < inputs[0] and right_bound > inputs[-1]):
                        raise RuntimeError("the inputs [%f, %f] of leaf node %d exceed the limits [%f, %f]" % (
                            inputs[0], inputs[-1], j, left_bound, right_bound))
                    inputs.insert(0, key_left_bounds[j])
                    inputs.append(key_left_bounds[j + 1] if j + 1 < task_size else 1 << self.geohash.sum_bits)
                    labels = list(range(0, len(inputs)))
                    # build model
                    pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple, is_gpu,
                                                weight, core, train_step, batch_num, learning_rate,
                                                use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(train_input[j], mp_list[j], None, None) for j in range(task_size)]
                # 2.2.1 create delta_index and delta_model
                for j in range(task_size):
                    # create the old_cdfs and old_max_keys for delta_model
                    node = nodes[j]
                    node.model.output_max -= 2  # remove the key bound
                    min_key = node.model.input_min
                    max_key = node.model.input_max
                    key_interval = (max_key - min_key) / cdf_width
                    key_list = [int(min_key + k * key_interval) for k in range(cdf_width)]
                    old_cdfs = [[] for k in range(self.cur_time_interval)]
                    for data in train_input[j]:
                        old_cdfs[(data[3] - start_time) // time_interval].append(data[2])
                    old_max_keys = [max(len(cdf) - 1, 0) for cdf in old_cdfs]
                    # for empty and head old_cdfs, remove them
                    l = 0
                    while l < self.cur_time_interval and len(old_cdfs[l]) == 0:
                        l += 1
                    old_cdfs = old_cdfs[l:]
                    for k in range(len(old_cdfs)):
                        cdf = old_cdfs[k]
                        if cdf:  # for non-empty old_cdfs, create by data
                            old_cdfs[k] = build_cdf(cdf, cdf_width, key_list)
                        else:  # for empty and non-head old_cdfs, copy from their previous
                            old_cdfs[k] = old_cdfs[k - 1]
                    # plot_ts(cdfs)
                    node.delta_model = TimeSeriesModel(old_cdfs, None, old_max_keys, None, key_list)
                    node.delta_model.build(cdf_width, cdf_lag)
                    node.delta_index = [[] for i in range(node.delta_model.cur_max_key + 1)]
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
        point = (point[0], point[1], gh, point[2], point[3])
        # 3. predict the leaf_node by rmi
        node_key = self.get_leaf_node(gh)
        # 4. insert ie into update index
        tg_list = self.get_delta_index_list(gh, self.rmi[-1][node_key])
        tg_list.insert(binary_search_less_max(tg_list, 2, gh, 0, len(tg_list) - 1) + 1, point)

    def insert(self, points):
        points = points.tolist()
        for point in points:
            cur_time = point[2]
            # update once the time of new point cross the time interval
            cur_time_interval = (cur_time - self.start_time) // self.time_interval
            if self.cur_time_interval < cur_time_interval:
                self.update()
                self.cur_time_interval = cur_time_interval
            self.insert_single(point)

    def update(self):
        """
        1. merge delta index into index
        2. update model
        3. update delta model
        :return:
        """
        leaf_nodes = self.rmi[-1]
        retrain_model_num = 0
        retrain_model_epoch = 0
        for j in range(0, self.stages[-1]):
            leaf_node = leaf_nodes[j]
            delta_index = []
            for tmp in leaf_node.delta_index:
                delta_index.extend(tmp)
            if delta_index:
                # 1. merge delta index into index
                if leaf_node.index:
                    leaf_node.index.extend(delta_index)
                    leaf_node.index.sort(key=lambda x: x[2])  # 优化：有序数组合并->sorted:2.5->1
                else:
                    leaf_node.index = delta_index
                # 2. update model
                inputs = [data[2] for data in leaf_node.index]
                inputs.insert(0, leaf_node.model.input_min)
                inputs.append(leaf_node.model.input_max)
                inputs_num = len(inputs)
                labels = list(range(0, inputs_num))
                batch_size = 2 ** math.ceil(math.log(inputs_num / self.batch_num, 2))
                if batch_size < 1:
                    batch_size = 1
                model_key = "retrain_%s" % j
                tmp_index = NN(self.model_path, model_key, inputs, labels, True, self.is_gpu, self.weight,
                               self.cores, self.train_step, batch_size, self.learning_rate, False, None, None)
                # tmp_index.train_simple(None)  # retrain with initial model
                tmp_index.build_simple(leaf_node.model.matrices if leaf_node.model else None)  # retrain with old model
                leaf_node.model = AbstractNN(tmp_index.get_matrices(), leaf_node.model.hl_nums,
                                             leaf_node.model.input_min, leaf_node.model.input_max,
                                             0, inputs_num - 3,
                                             math.ceil(tmp_index.min_err), math.ceil(tmp_index.max_err))
                retrain_model_num += 1
                retrain_model_epoch += tmp_index.get_epochs()
                # 3. update delta model
                leaf_node.delta_model.update([data[2] for data in delta_index], self.cdf_width, self.cdf_lag)
                leaf_node.delta_index = [[] for i in range(leaf_node.delta_model.cur_max_key + 1)]
        self.logging.info("Retrain model num: %s" % retrain_model_num)
        self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)

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

    def get_delta_index_list(self, key, leaf_node):
        """
        get the delta_index list which contains the key
        :param key: float
        :param leaf_node: node
        :return: the list of delta_index
        """
        pos = (key - leaf_node.model.input_min) / (
                leaf_node.model.input_max - leaf_node.model.input_min) * self.cdf_width
        pos_int = int(pos)
        left_p = leaf_node.delta_model.cur_cdf[pos_int]
        if pos >= self.cdf_width - 1:  # if point is at the top of cdf(1.0), insert into the tail of delta_index
            key = leaf_node.delta_model.cur_max_key
        else:
            right_p = leaf_node.delta_model.cur_cdf[pos_int + 1]
            key = int((left_p + (right_p - left_p) * (pos - pos_int)) * leaf_node.delta_model.cur_max_key)
        return leaf_node.delta_index[key]

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

    def get_leaf_bound(self):
        """
        get the key bound of leaf node by rmi
        1. use non-leaf rmi as func
        2. get the key bound for per leaf node
        e.g. rmi(x-1)=i-1, rmi(x)=i, rmi(y-1)=i, rmi(y)=i+1, so key_bound_i=(x, y),
        """
        leaf_node_len = self.stages[-1]
        key_left_bounds = [0]
        left = 0
        max_key = 1 << self.geohash.sum_bits
        for i in range(1, leaf_node_len):
            right = max_key
            while left <= right:
                mid = (left + right) >> 1
                if self.get_leaf_node(mid) >= i:
                    if self.get_leaf_node(mid - 1) >= i:
                        right = mid - 1
                    else:
                        key_left_bounds.append(mid)
                        break
                else:
                    left = mid + 1
        return key_left_bounds

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
        result = [leaf_node.index[key][4] for key in biased_search(leaf_node.index, 2, gh, pre, l_bound, r_bound)]
        # 4. filter in update index
        if leaf_node.delta_index:
            tg_list = self.get_delta_index_list(gh, leaf_node)
            result.extend([tg_list[key][4] for key in binary_search(tg_list, 2, gh, 0, len(tg_list) - 1)])
        return result

    def save(self):
        meta = np.array((self.geohash.data_precision,
                         self.geohash.region.bottom, self.geohash.region.up,
                         self.geohash.region.left, self.geohash.region.right,
                         self.is_gpu, self.weight, self.train_step, self.batch_num, self.learning_rate,
                         self.start_time, self.cur_time_interval, self.time_interval,
                         self.cdf_lag, self.cdf_width),
                        dtype=[("0", 'i4'),
                               ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8'),
                               ("5", 'i1'), ("6", 'f4'), ("7", 'i2'), ("8", 'i2'), ("9", 'f4'),
                               ("10", 'i4'), ("11", 'i4'), ("12", 'i4'),
                               ("13", 'i1'), ("14", 'i1')])
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
        delta_models = []
        for node in self.rmi[-1]:
            indexes.extend(node.index)
            index_lens.append(len(node.index))
            delta_index = []
            for tmp in node.delta_index:
                delta_index.extend(tmp)
            delta_indexes.extend(delta_index)
            delta_index_lens.append(len(delta_index))
            delta_models.append(node.delta_model)
        np.save(os.path.join(self.model_path, 'indexes.npy'),
                np.array(indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'index_lens.npy'), index_lens)
        np.save(os.path.join(self.model_path, 'delta_indexes.npy'),
                np.array(delta_indexes, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i8'), ("3", 'i4'), ("4", 'i4')]))
        np.save(os.path.join(self.model_path, 'delta_index_lens.npy'), delta_index_lens)
        np.save(os.path.join(self.model_path, 'delta_models.npy'), delta_models)

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
        self.start_time = meta[10]
        self.cur_time_interval = meta[11]
        self.time_interval = meta[12]
        self.cdf_lag = meta[13]
        self.cdf_width = meta[14]
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
                self.rmi.append(
                    [Node(None, model, None, None) for model in models[model_cur:model_cur + self.stages[i]]])
                model_cur += self.stages[i]
            else:
                index_cur = 0
                delta_index_cur = 0
                leaf_nodes = []
                for j in range(self.stages[i]):
                    model = models[model_cur]
                    delta_index = delta_indexes[delta_index_cur:delta_index_cur + delta_index_lens[j]]
                    delta_model = delta_models[j]
                    delta_index_lists = [[] for i in range(delta_model.cur_max_key + 1)]
                    for tmp in delta_index:
                        pos = (tmp[2] - model.input_min) / (model.input_max - model.input_min) * self.cdf_width
                        pos_int = int(pos)
                        left_p = delta_model.cur_cdf[pos_int]
                        if pos >= self.cdf_width - 1:  # if point is at the top of cdf(1.0), insert into the tail of delta_index
                            key = delta_model.cur_max_key
                        else:
                            right_p = delta_model.cur_cdf[pos_int + 1]
                            key = int((left_p + (right_p - left_p) * (pos - pos_int)) * delta_model.cur_max_key)
                        delta_index_lists[key].append(tmp)
                    leaf_nodes.append(Node(indexes[index_cur:index_cur + index_lens[j]],
                                           models[model_cur],
                                           delta_index_lists,
                                           delta_model))
                    model_cur += 1
                    index_cur += index_lens[j]
                    delta_index_cur += delta_index_lens[j]
                self.rmi.append(leaf_nodes)

    def size(self):
        """
        structure_size = meta.npy + stages.npy + cores.npy + models.npy
        ie_size = index_lens.npy + indexes.npy + delta_index_lens.npy + delta_indexes.npy
        """
        return os.path.getsize(os.path.join(self.model_path, "meta.npy")) - 128 - 64 * 3 + \
               os.path.getsize(os.path.join(self.model_path, "stages.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "cores.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "models.npy")) - 128, \
               os.path.getsize(os.path.join(self.model_path, "index_lens.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "indexes.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "delta_index_lens.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "delta_indexes.npy")) - 128

    def model_err(self):
        model_precisions = [(node.model.max_err - node.model.min_err) for node in self.rmi[-1] if node.model]
        model_precisions_avg = sum(model_precisions) / self.stages[-1]
        return model_precisions_avg

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
    tmp_index.build()
    abstract_index = AbstractNN(tmp_index.get_matrices(), len(core) - 1,
                                int(tmp_index.train_x_min), int(tmp_index.train_x_max),
                                int(tmp_index.train_y_min), int(tmp_index.train_y_max),
                                math.ceil(tmp_index.min_err), math.ceil(tmp_index.max_err))
    del tmp_index
    gc.collect(generation=0)
    mp_list[current_stage_step] = abstract_index


class Node:
    def __init__(self, index, model, delta_index, delta_model):
        self.index = index
        self.model = model
        self.delta_index = delta_index
        self.delta_model = delta_model


class NN(MLP):
    def __init__(self, model_path, model_key, train_x, train_y, is_new, is_gpu, weight, core, train_step, batch_size,
                 learning_rate, use_threshold, threshold, retrain_time_limit):
        self.name = "MULIS NN"
        # train_x的是有序的，归一化不需要计算最大最小值
        train_x_min = train_x[0]
        train_x_max = train_x[-1]
        train_x = (np.array(train_x) - train_x_min) / (train_x_max - train_x_min) - 0.5
        train_y_min = train_y[0]
        train_y_max = train_y[-1]
        train_y = (np.array(train_y) - train_y_min) / (train_y_max - train_y_min)
        super().__init__(model_path, model_key, train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         is_new, is_gpu, weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                         retrain_time_limit)


class NNSimple(MLPSimple):
    def __init__(self, train_x, train_y, is_gpu, weight, core, train_step, batch_size, learning_rate):
        self.name = "MULIS NN"
        # train_x的是有序的，归一化不需要计算最大最小值
        train_x_min = train_x[0]
        train_x_max = train_x[-1]
        train_x = (np.array(train_x) - train_x_min) / (train_x_max - train_x_min) - 0.5
        train_y_min = train_y[0]
        train_y_max = train_y[-1]
        train_y = (np.array(train_y) - train_y_min) / (train_y_max - train_y_min)
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


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/mulis_10w_nyct/"
    data_distribution = Distribution.NYCT_10W
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = Mulis(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list,
                    is_sorted=False,
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
                    use_thresholds=[False, False],
                    learning_rates=[0.001, 0.001],
                    thresholds=[5, 20],
                    retrain_time_limits=[4, 2],
                    thread_pool_size=6,
                    time_interval=60 * 60 * 24,
                    start_time=1356998400,
                    end_time=1359676799,
                    cdf_width=100,
                    cdf_lag=3)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    logging.info("Model precision avg: %s" % index.model_err())
    point_query_list = load_query(data_distribution, 0).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)[:1000]
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))
    index.save()
    index.load()


if __name__ == '__main__':
    main()
