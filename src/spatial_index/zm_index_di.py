import copy
import logging
import math
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/SLBRIN')
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.spatial_index.zm_index_optimised import ZMIndexOptimised, NN

# 预设pagesize=4096, size(model)=2000, size(pointer)=4, size(x/y/geohash)=8
PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class ZMIndexDeltaInsert(ZMIndexOptimised):
    def __init__(self, model_path=None):
        super(ZMIndexDeltaInsert, self).__init__(model_path)
        # for update
        self.start_time = None
        self.time_id = None
        self.time_interval = None
        # for compute
        self.is_retrain = None
        self.is_save = None
        self.statistic_list = [0, 0, 0, 0, 0]  # insert_key time and io, merge_data time and io, retrain_model time

    def build_append(self, time_interval, start_time, end_time, is_retrain, is_save):
        self.start_time = start_time
        self.time_id = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
        self.is_retrain = is_retrain
        self.is_save = is_save

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
            self.statistic_list[0] += time.time() - start_time
            self.statistic_list[1] += self.io_cost - io_cost

    def update(self):
        """
         update the whole index
         1. merge delta index into index
         2. update model
         """
        retrain_model_num = 0
        retrain_model_epoch = 0
        retrain_model_time = self.statistic_list[4]
        # 1. merge delta index into index
        update_list = [0] * self.stages[-1]
        for j in range(self.stages[-1]):
            leaf_node = self.rmi[-1][j]
            if leaf_node.delta_index:
                update_list[j] = 1
                start_time = time.time()
                io_cost = self.io_cost
                if leaf_node.index:
                    leaf_node.index.extend(leaf_node.delta_index)
                    leaf_node.index.sort(key=lambda x: x[2])  # 优化：有序数组合并->sorted:2.5->1
                else:
                    leaf_node.index = leaf_node.delta_index
                leaf_node.delta_index = []
                self.statistic_list[2] += time.time() - start_time
                # IO1: merge data
                self.io_cost += math.ceil(len(leaf_node.index) / ITEMS_PER_PAGE)
                self.statistic_list[3] += self.io_cost - io_cost
        # 2. update model
        if self.is_retrain and self.time_id > 763:
            start_time = time.time()
            pool = multiprocessing.Pool(processes=4)
            mp_dict = multiprocessing.Manager().dict()
            for j in range(0, self.stages[-1]):
                if update_list[j] == 1:
                    leaf_node = self.rmi[-1][j]
                    pool.apply_async(update_leaf_node,
                                     (self.model_path, j, leaf_node.index, leaf_node.model, self.weight, self.cores, self.train_step,
                                      self.batch_num, self.learning_rate, mp_dict))
            pool.close()
            pool.join()
            for (key, value) in mp_dict.items():
                self.rmi[-1][key].model = value[0]
                retrain_model_num += value[1]
                retrain_model_epoch += value[2]
            self.logging.info("Retrain model num: %s" % retrain_model_num)
            self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)
            self.logging.info("Retrain model time: %s" % (time.time() - start_time))
        else:
            time_model_path = os.path.join(self.model_path, "../zm_time_model", str(self.time_id))
            index = ZMIndexOptimised(model_path=time_model_path)
            index.load()
            for j in range(0, self.stages[-1]):
                if update_list[j] == 1:
                    self.rmi[-1][j].model = index.rmi[-1][j].model
        if self.is_save:
            time_model_path = os.path.join(self.model_path, "../zm_time_model", str(self.time_id))
            if os.path.exists(time_model_path) is False:
                os.makedirs(time_model_path)
            index = copy.deepcopy(self)
            index.model_path = time_model_path
            index.save()

    def save(self):
        super(ZMIndexDeltaInsert, self).save()
        meta_append = np.array((self.start_time, self.time_id, self.time_interval),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4')])
        np.save(os.path.join(self.model_path, 'meta_append.npy'), meta_append)

    def load(self):
        super(ZMIndexDeltaInsert, self).load()
        meta_append = np.load(os.path.join(self.model_path, 'meta_append.npy'), allow_pickle=True).item()
        self.start_time = meta_append[0]
        self.time_id = meta_append[1]
        self.time_interval = meta_append[2]

    def size(self):
        structure_size, ie_size = super(ZMIndexDeltaInsert, self).size()
        structure_size += os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128
        return structure_size, ie_size


def update_leaf_node(model_path, model_key, inputs, model, weight, cores, train_step, batch_num,
                     learning_rate, mp_dict):
    inputs = [data[2] for data in inputs]
    inputs.insert(0, model.input_min)
    inputs.append(model.input_max)
    inputs_num = len(inputs)
    labels = list(range(0, inputs_num))
    batch_size = 2 ** math.ceil(math.log(inputs_num / batch_num, 2))
    if batch_size < 1:
        batch_size = 1
    tmp_index = NN(model_path, model_key, inputs, labels, True, weight,
                   cores, train_step, batch_size, learning_rate, False, None, None)
    # tmp_index.build_simple(None)  # retrain with initial model
    tmp_index.build_simple(model.matrices if model else None)  # retrain with old model
    model.matrices = tmp_index.get_matrices()
    model.output_max = inputs_num - 3
    model.max_err = math.ceil(tmp_index.max_err)
    mp_dict[model_key] = (model, 1, tmp_index.get_epochs())


def main():
    load_index_from_json = True
    load_index_from_json2 = False
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/zmdi_10w_nyct/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = ZMIndexDeltaInsert(model_path=model_path)
    index_name = index.name
    if load_index_from_json:
        super(ZMIndexDeltaInsert, index).load()
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
                           is_retrain=True,
                           is_save=False)
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
    logging.info("Statis list: %s" % index.statistic_list)
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
