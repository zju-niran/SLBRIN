import logging
import math
import multiprocessing
import os
import time

import numpy as np

from src.utils.common_utils import biased_search_less_max_duplicate, biased_search_duplicate
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.proposed_sli.slibs import SLIBS
from src.proposed_sli.dtusli import retrain_model

PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class IPUSLI(SLIBS):
    """
    就地更新空间学习型索引（In-place Update Spatial Learned Index，IPUSLI）
    1. 基本思路：在SLIBS的基础上应用FITing-tree（Fiting-tree: A data-aware index structure）的就地更新方法（In-place Update Method，IPUM）
    """
    def __init__(self, model_path=None):
        super(IPUSLI, self).__init__(model_path)
        # for update
        self.start_time = None
        self.time_id = None
        self.time_interval = None
        self.empty_ratio = None
        self.is_init = False
        self.threshold_err = 1
        # for compute
        self.is_retrain = True
        self.time_retrain = -1
        self.thread_retrain = 1
        self.is_save = True
        self.insert_time = 0
        self.insert_io = 0
        self.last_insert_time = 0
        self.last_insert_io = 0

    def expand_leaf_node(self, leaf_node):
        """
        preserve the empty locations at the beginning and end of index
        expand if too few, or reduce if too many
        """
        # 1. compute the len of empty locations = empty_ratio * err_bound
        model = leaf_node.model
        model_err = model.max_err - model.min_err
        empty_len = max(1, int(self.empty_ratio * model_err))
        # 2. get current len of empty locations
        cur_left_len = model.bias
        cur_right_len = len(leaf_node.index) - model.output_max - cur_left_len - 1
        # 3. keep the len of empty locations is equal to empty_len
        diff_left_len = empty_len - cur_left_len
        diff_right_len = empty_len - cur_right_len
        # if they are too few, expand them as [empty, old_index, empty]
        # else if they are too many, reduce them
        if diff_left_len > 0:
            index = [(0, 0, 0, 0, 0) for i in range(diff_left_len)]
            index.extend(leaf_node.index)
        else:
            index = leaf_node.index[-diff_left_len:]
        if diff_right_len >= 0:
            index.extend([(0, 0, 0, 0, 0) for i in range(diff_right_len)])
        else:
            index = index[:diff_right_len]
        if diff_left_len != 0:  # the move of left empty locations cost IO
            self.io_cost += math.ceil(model.output_max / ITEMS_PER_PAGE)
        leaf_node.index = index
        # 4. update the bias and error bound of model
        model.bias = empty_len
        model.min_err -= empty_len
        model.max_err += empty_len
        return diff_left_len

    def build_append(self, time_interval, start_time, end_time, empty_ratio, is_init, threshold_err,
                     is_retrain, time_retrain, thread_retrain, is_save):
        """
        1. preserve empty locations for each leaf node
        state: need to update when state = 1
        bias: the len of empty locations in the left, index = [empty locations with len of bias,
                                                                data with len of output_max,
                                                                empty locations]
        """
        self.start_time = start_time
        self.time_id = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
        self.empty_ratio = empty_ratio
        self.is_init = is_init
        self.threshold_err = threshold_err
        self.is_retrain = is_retrain
        self.time_retrain = time_retrain
        self.thread_retrain = thread_retrain
        self.is_save = is_save
        # 1. preserve empty locations for each leaf node
        for leaf_node in self.rmi[-1]:
            leaf_node.model.state = 0  # state: need to update when state = 1
            leaf_node.model.bias = 0  # index = [empty]
            self.expand_leaf_node(leaf_node)

    def insert_single(self, point):
        """
        different from zm_index
        1. insert into index instead of delta_index
        2. move the relative data towards the closest direction
        3. insert key into the empty location
        """
        # 1. find the key of point by point query
        gh = self.geohash.encode(point[0], point[1])
        point = (point[0], point[1], gh, point[2], point[3])
        # insert into index instead of delta_index
        leaf_node, _, pre, min_err, max_err = self.predict(gh)
        model = leaf_node.model
        # update the state of model
        model.state = 1
        # rectify the search bound with bias
        l_bound = max(pre - max_err, model.output_min) + model.bias
        r_bound = min(pre - min_err, model.output_max) + model.bias
        pre += model.bias
        key = biased_search_less_max_duplicate(leaf_node.index, 2, gh, pre, l_bound, r_bound)
        # IO1: search key
        self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
        # 2. insert point at the key
        # move the relative data towards the closest direction
        # IO2: move data when inserting key
        if key - model.bias >= model.bias + model.output_max - key:
            if model.bias + model.output_max >= len(leaf_node.index) - 1:
                # IO3: move data when expanding data
                key += self.expand_leaf_node(leaf_node)
            for i in range(model.output_max + model.bias, key - 1, -1):
                leaf_node.index[i + 1] = leaf_node.index[i]
            model.output_max += 1
            # insert key into the empty location
            leaf_node.index[key] = point
            self.io_cost += math.ceil((model.output_max + model.bias - key + 1) / ITEMS_PER_PAGE)
        else:
            if model.bias <= 0:
                key += self.expand_leaf_node(leaf_node)
            for i in range(model.bias, key):
                leaf_node.index[i - 1] = leaf_node.index[i]
            model.bias -= 1
            model.output_max += 1
            leaf_node.index[key - 1] = point
            self.io_cost += math.ceil((key - model.bias) / ITEMS_PER_PAGE)

    def insert(self, points):
        """
        different from zm_index
        1. update once the time of new point cross the time interval
        """
        points = points.tolist()
        for point in points:
            cur_time = point[2]
            # 1. update once the time of new point cross the time interval
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
        1. update model
        """
        self.logging.info("Update time id: %s" % self.time_id)
        self.logging.info("Insert key time: %s" % (self.insert_time - self.last_insert_time))
        self.logging.info("Insert key io: %s" % (self.insert_io - self.last_insert_io))
        self.last_insert_time = self.insert_time
        self.last_insert_io = self.insert_io
        # 1. update model
        retrain_model_num = 0
        retrain_model_epoch = 0
        retrain_model_time = 0
        retrain_model_io = 0
        if self.is_retrain and self.time_id > self.time_retrain:
            start_time = time.time()
            pool = multiprocessing.Pool(processes=self.thread_retrain)
            mp_dict = multiprocessing.Manager().dict()
            for j in range(0, self.stages[-1]):
                leaf_node = self.rmi[-1][j]
                model = leaf_node.model
                if model.state != 0:
                    inputs = leaf_node.index[model.bias:model.bias + model.output_max + 1]
                    retrain_model_io += math.ceil(len(inputs) / ITEMS_PER_PAGE)
                    pool.apply_async(retrain_model,
                                     (self.model_path, j, inputs, model,
                                      self.weight, self.cores, self.train_step, self.batch_num, self.learning_rate,
                                      self.is_init, self.threshold_err, mp_dict))
            pool.close()
            pool.join()
            for (key, value) in mp_dict.items():
                leaf_node = self.rmi[-1][key]
                state = leaf_node.model.state
                bias = leaf_node.model.bias
                leaf_node.model = value[0]
                leaf_node.model.state = state
                leaf_node.model.bias = bias
                retrain_model_num += value[1]
                retrain_model_epoch += value[2]
            retrain_model_time = time.time() - start_time
        else:
            time_model_path = os.path.join(self.model_path, "../sli_time_model_model", str(self.time_id), 'models.npy')
            models = np.load(time_model_path, allow_pickle=True)[-self.stages[-1]:]
            for j in range(self.stages[-1]):
                leaf_node = self.rmi[-1][j]
                state = leaf_node.model.state
                bias = leaf_node.model.bias
                leaf_node.model = models[j]
                leaf_node.model.state = state
                leaf_node.model.bias = bias
        if self.is_save:
            time_model_path = os.path.join(self.model_path, "../sli_time_model_model", str(self.time_id))
            if os.path.exists(time_model_path) is False:
                os.makedirs(time_model_path)
            models = []
            for stage in self.rmi:
                for node in stage:
                    model = node.model
                    del model.state
                    del model.bias
                    models.append(model)
            np.save(os.path.join(time_model_path, 'models.npy'), models)
        start_io = self.io_cost
        start_time = time.time()
        index_len = 0
        for j in range(0, self.stages[-1]):
            leaf_node = self.rmi[-1][j]
            if leaf_node.model.state != 0:
                # 2. ensure the len of empty locations is enough
                self.expand_leaf_node(leaf_node)
                # 3. reset the state of model
                leaf_node.model.state = 0
            index_len += len(leaf_node.index)
        self.logging.info("Merge data time: %s" % (time.time() - start_time))
        self.logging.info("Merge data io: %s" % (self.io_cost - start_io))
        if self.is_retrain and self.time_id > self.time_retrain:
            self.logging.info("Retrain model num: %s" % retrain_model_num)
            self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)
            self.logging.info("Retrain model time: %s" % retrain_model_time)
            self.logging.info("Retrain model io: %s" % retrain_model_io)
        self.logging.info("Index entry size: %s" % (index_len * ITEM_SIZE))
        self.logging.info("Error bound: %s" % self.model_err())

    def point_query_single(self, point):
        """
        different from zm_index
        1. rectify the search bound with bias
        """
        gh = self.geohash.encode(point[0], point[1])
        leaf_node, _, pre, min_err, max_err = self.predict(gh)
        model = leaf_node.model
        # 1. rectify the search bound with bias
        l_bound = max(pre - max_err, model.output_min) + model.bias
        r_bound = min(pre - min_err, model.output_max) + model.bias
        pre += model.bias
        result = [leaf_node.index[key][4] for key in
                  biased_search_duplicate(leaf_node.index, 2, gh, pre, l_bound, r_bound)]
        self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
        return result

    def save(self):
        super(IPUSLI, self).save()
        meta_append = np.array((self.start_time, self.time_id, self.time_interval,
                                self.empty_ratio, self.is_init, self.threshold_err),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'),
                                      ("3", 'f8'), ("4", 'i1'), ("5", 'f8')])
        np.save(os.path.join(self.model_path, 'meta_append.npy'), meta_append)
        compute = np.array((self.is_retrain, self.time_retrain, self.thread_retrain, self.is_save),
                           dtype=[("0", 'i1'), ("1", 'i2'), ("2", 'i1'), ("3", 'i1')])
        np.save(os.path.join(self.model_path, 'compute.npy'), compute)

    def load(self):
        super(IPUSLI, self).load()
        meta_append = np.load(os.path.join(self.model_path, 'meta_append.npy'), allow_pickle=True).item()
        self.start_time = meta_append[0]
        self.time_id = meta_append[1]
        self.time_interval = meta_append[2]
        self.empty_ratio = meta_append[3]
        self.is_init = bool(meta_append[4])
        self.threshold_err = meta_append[5]
        compute = np.load(os.path.join(self.model_path, 'compute.npy'), allow_pickle=True).item()
        self.is_retrain = bool(compute[0])
        self.time_retrain = compute[1]
        self.thread_retrain = compute[2]
        self.is_save = bool(compute[3])

    def size(self):
        """
        structure_size += meta_append.npy
        ie_size = index
        """
        index_len = 0
        for leaf_node in self.rmi[-1]:
            index_len += len(leaf_node.index)
        return os.path.getsize(os.path.join(self.model_path, "meta.npy")) - 128 - 64 * 2 + \
               os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "stages.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "cores.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "models.npy")) - 128, \
               index_len * ITEM_SIZE


def main():
    load_index_from_json = True
    load_index_from_json2 = False
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/ipusli_10w_nyct/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = IPUSLI(model_path=model_path)
    index_name = index.name
    if load_index_from_json:
        super(IPUSLI, index).load()
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
                           empty_ratio=0.5,
                           is_init=True,
                           threshold_err=0,
                           is_retrain=True,
                           time_retrain=-1,
                           thread_retrain=3,
                           is_save=False)
        # index.save()
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
