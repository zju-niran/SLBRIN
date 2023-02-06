import logging
import math
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/SLBRIN')
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.spatial_index.common_utils import biased_search_duplicate, binary_search_less_max, binary_search_duplicate, \
    Region
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.zm_index import Node, AbstractNN
from src.spatial_index.zm_index_optimised import ZMIndexOptimised, NN
from src.ts_model import TimeSeriesModel, build_cdf

# 预设pagesize=4096, size(model)=2000, size(pointer)=4, size(x/y/geohash)=8
PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class Mulis(ZMIndexOptimised):
    def __init__(self, model_path=None):
        super(Mulis, self).__init__(model_path)
        # 更新所需：
        self.start_time = None
        self.cur_time_interval = None
        self.time_interval = None
        self.cdf_width = None
        self.cdf_lag = None

    def build_append(self, time_interval, start_time, end_time, cdf_width, cdf_lag):
        """
        1. create delta_model with ts_model
        2. change delta_index from [] into [[]]
        """
        self.start_time = start_time
        self.cur_time_interval = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
        self.cdf_width = cdf_width
        self.cdf_lag = cdf_lag
        # 1. create delta_model with ts_model
        for j in range(self.stages[-1]):
            node = self.rmi[-1][j]
            # create the old_cdfs and old_max_keys for delta_model
            min_key = node.model.input_min
            max_key = node.model.input_max
            key_interval = (max_key - min_key) / self.cdf_width
            key_list = [int(min_key + k * key_interval) for k in range(self.cdf_width)]
            old_cdfs = [[] for k in range(self.cur_time_interval)]
            for data in node.index:
                old_cdfs[(data[3] - self.start_time) // self.time_interval].append(data[2])
            old_max_keys = [max(len(cdf) - 1, 0) for cdf in old_cdfs]
            # for empty and head old_cdfs, remove them
            l = 0
            while l < self.cur_time_interval and len(old_cdfs[l]) == 0:
                l += 1
            old_cdfs = old_cdfs[l:]
            for k in range(len(old_cdfs)):
                cdf = old_cdfs[k]
                if cdf:  # for non-empty old_cdfs, create by data
                    old_cdfs[k] = build_cdf(cdf, self.cdf_width, key_list)
                else:  # for empty and non-head old_cdfs, copy from their previous
                    old_cdfs[k] = old_cdfs[k - 1]
            # plot_ts(cdfs)
            node.delta_model = TimeSeriesModel(old_cdfs, None, old_max_keys, None, key_list)
            node.delta_model.build(self.cdf_width, self.cdf_lag)
            # 2. change delta_index from list into list of list
            node.delta_index = [[] for i in range(node.delta_model.cur_max_key + 1)]

    def insert_single(self, point):
        """
        different from zm_index
        1. find and insert ie into the target list of delta_index
        """
        gh = self.geohash.encode(point[0], point[1])
        point = (point[0], point[1], gh, point[2], point[3])
        node_key = self.get_leaf_node(gh)
        # 1. find and insert ie into the target list of delta_index
        tg_list = self.get_delta_index_list(gh, self.rmi[-1][node_key])
        tg_list.insert(binary_search_less_max(tg_list, 2, gh, 0, len(tg_list) - 1) + 1, point)

    def insert(self, points):
        """
        different from zm_index
        1. update once the time of new point cross the time interval
        """
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
        update the whole index
        1. merge delta index into index
        2. update model
        3. update delta model
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
                                             math.floor(tmp_index.min_err), math.ceil(tmp_index.max_err))
                retrain_model_num += 1
                retrain_model_epoch += tmp_index.get_epochs()
                # 3. update delta model
                leaf_node.delta_model.update([data[2] for data in delta_index], self.cdf_width, self.cdf_lag)
                leaf_node.delta_index = [[] for i in range(leaf_node.delta_model.cur_max_key + 1)]
        self.logging.info("Retrain model num: %s" % retrain_model_num)
        self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)

    def get_delta_index_list(self, key, leaf_node):
        """
        get the delta_index list which contains the key
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
        # 1. find the target list of delta_index which contains the target ie
        if leaf_node.delta_index:
            tg_list = self.get_delta_index_list(gh, leaf_node)
            result.extend([tg_list[key][4] for key in binary_search_duplicate(tg_list, 2, gh, 0, len(tg_list) - 1)])
        return result

    def save(self):
        """
        different from zm_index
        1. the delta_index isn't list but a list of list
        """
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
        """
        different from zm_index
        1. the delta_index isn't list but a list of list
        """
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
        structure_size, ie_size = super(Mulis, self).size()
        structure_size += os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128
        ie_size += os.path.getsize(os.path.join(self.model_path, "delta_models.npy")) - 128
        return structure_size, ie_size


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/mulis_10w_nyct/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = Mulis(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
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
                    is_gpu=True,
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
        index.build_append(time_interval=60 * 60 * 24,
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
    io_cost = 0
    logging.info("Model precision avg: %s" % index.model_err())
    point_query_list = load_query(data_distribution, 0).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Update time: %s" % (end_time - start_time))
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
