import logging
import math
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/SLBRIN')
from src.spatial_index.common_utils import biased_search_less_max_duplicate, biased_search_duplicate, \
    binary_search_duplicate
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.spatial_index.zm_index_optimised import ZMIndexOptimised, NN

# 预设pagesize=4096, size(model)=2000, size(pointer)=4, size(x/y/geohash)=8
PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class ZMIndexInPlaceInsert(ZMIndexOptimised):
    def __init__(self, model_path=None):
        super(ZMIndexInPlaceInsert, self).__init__(model_path)
        # 更新所需：
        self.start_time = None
        self.cur_time_interval = None
        self.time_interval = None
        self.empty_ratio = None

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
        leaf_node.index = index
        # 4. update the bias and error bound of model
        model.bias = empty_len
        model.min_err -= empty_len
        model.max_err += empty_len
        return diff_left_len

    def build_append(self, time_interval, start_time, end_time, empty_ratio):
        """
        1. preserve empty locations for each leaf node
        state: need to update when state = 1
        bias: the len of empty locations in the left, index = [empty locations with len of bias,
                                                                data with len of output_max,
                                                                empty locations]
        """
        self.start_time = start_time
        self.cur_time_interval = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
        self.empty_ratio = empty_ratio
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
        gh = self.geohash.encode(point[0], point[1])
        point = (point[0], point[1], gh, point[2], point[3])
        # 1. insert into index instead of delta_index
        leaf_node, node_key, pre, min_err, max_err = self.predict(gh)
        model = leaf_node.model
        # update the state of model
        model.state = 1
        # rectify the search bound with bias
        l_bound = max(pre - max_err, model.output_min) + model.bias
        r_bound = min(pre - min_err, model.output_max) + model.bias
        pre += model.bias
        key = biased_search_less_max_duplicate(leaf_node.index, 2, gh, pre, l_bound, r_bound)
        # 2. move the relative data towards the closest direction
        if key - model.bias >= model.bias + model.output_max - key:
            if model.output_max + model.bias >= len(leaf_node.index) - 1:
                key += self.expand_leaf_node(leaf_node)
            for i in range(model.output_max + model.bias, key - 1, -1):
                leaf_node.index[i + 1] = leaf_node.index[i]
            model.output_max += 1
            # 3. insert key into the empty location
            leaf_node.index[key] = point
        else:
            if model.bias <= 0:
                key += self.expand_leaf_node(leaf_node)
            for i in range(model.bias, key):
                leaf_node.index[i - 1] = leaf_node.index[i]
            model.bias -= 1
            model.output_max += 1
            leaf_node.index[key - 1] = point

    def insert(self, points):
        """
        different from zm_index
        1. update once the time of new point cross the time interval
        """
        points = points.tolist()
        for point in points:
            cur_time = point[2]
            # 1. update once the time of new point cross the time interval
            cur_time_interval = (cur_time - self.start_time) // self.time_interval
            if self.cur_time_interval < cur_time_interval:
                self.update()
                self.cur_time_interval = cur_time_interval
            self.insert_single(point)

    def update(self):
        """
        update the whole index
        1. update model
        """
        leaf_nodes = self.rmi[-1]
        retrain_model_num = 0
        retrain_model_epoch = 0
        for j in range(0, self.stages[-1]):
            leaf_node = leaf_nodes[j]
            model = leaf_node.model
            if model.state != 0:
                # 1. update model
                inputs = [data[2] for data in leaf_node.index[model.bias:model.bias + model.output_max + 1]]
                inputs.insert(0, model.input_min)
                inputs.append(model.input_max)
                inputs_num = len(inputs)
                labels = list(range(0, inputs_num))
                batch_size = 2 ** math.ceil(math.log(inputs_num / self.batch_num, 2))
                if batch_size < 1:
                    batch_size = 1
                model_key = "retrain_%s" % j
                tmp_index = NN(self.model_path, model_key, inputs, labels, True, self.is_gpu, self.weight,
                               self.cores, self.train_step, batch_size, self.learning_rate, False, None, None)
                # tmp_index.train_simple(None)  # retrain with initial model
                tmp_index.build_simple(model.matrices if model else None)  # retrain with old model
                model.matrices = tmp_index.get_matrices()
                model.min_err = math.floor(tmp_index.min_err)
                model.max_err = math.ceil(tmp_index.max_err)
                # 2. ensure the len of empty locations is enough
                self.expand_leaf_node(leaf_node)
                # 3. reset the state of model
                model.state = 0
                retrain_model_num += 1
                retrain_model_epoch += tmp_index.get_epochs()
                del tmp_index
        self.logging.info("Retrain model num: %s" % retrain_model_num)
        self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)

    def point_query_single(self, point):
        """
        different from zm_index
        1. rectify the search bound with bias
        """
        gh = self.geohash.encode(point[0], point[1])
        leaf_node, node_key, pre, min_err, max_err = self.predict(gh)
        model = leaf_node.model
        # 1. rectify the search bound with bias
        l_bound = max(pre - max_err, model.output_min) + model.bias
        r_bound = min(pre - min_err, model.output_max) + model.bias
        pre += model.bias
        result = [leaf_node.index[key][4] for key in
                  biased_search_duplicate(leaf_node.index, 2, gh, pre, l_bound, r_bound)]
        self.io_cost += math.ceil((r_bound - l_bound) / ITEMS_PER_PAGE)
        if leaf_node.delta_index:
            delta_index_len = len(leaf_node.delta_index)
            result.extend([leaf_node.delta_index[key][4]
                           for key in
                           binary_search_duplicate(leaf_node.delta_index, 2, gh, 0, len(leaf_node.delta_index) - 1)])
            self.io_cost += delta_index_len // ITEMS_PER_PAGE + 1
        return result

    def save(self):
        super(ZMIndexInPlaceInsert, self).save()
        meta_append = np.array((self.start_time, self.cur_time_interval, self.time_interval, self.empty_ratio),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'), ("3", 'f8')])
        np.save(os.path.join(self.model_path, 'meta_append.npy'), meta_append)

    def load(self):
        super(ZMIndexInPlaceInsert, self).load()
        meta_append = np.load(os.path.join(self.model_path, 'meta_append.npy'), allow_pickle=True).item()
        self.start_time = meta_append[0]
        self.cur_time_interval = meta_append[1]
        self.time_interval = meta_append[2]
        self.empty_ratio = meta_append[3]

    def size(self):
        return os.path.getsize(os.path.join(self.model_path, "meta.npy")) - 128 - 64 * 2 + \
               os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "stages.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "cores.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "models.npy")) - 128, \
               os.path.getsize(os.path.join(self.model_path, "index_lens.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "indexes.npy")) - 128


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/zmipi_10w_nyct/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = ZMIndexInPlaceInsert(model_path=model_path)
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
                    thread_pool_size=4)
        index.build_append(time_interval=60 * 60 * 24,
                           start_time=1356998400,
                           end_time=1359676799,
                           empty_ratio=0.5)
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
