import logging
import math
import multiprocessing
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.proposed_sli.slibs import SLIBS, NN
from src.sli.zm_index import Array

PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class DTUSLI(SLIBS):
    """
    增量更新空间学习型索引（Delta Update Spatial Learned Index，DTUSLI）
    1. 基本思路：在SLIBS的基础上应用FITing-tree（Fiting-tree: A data-aware index structure）的增量更新方法（Delta Update Method，DTUM）
    """
    def __init__(self, model_path=None):
        super(DTUSLI, self).__init__(model_path)
        # for update
        self.start_time = None
        self.time_id = None
        self.time_interval = None
        self.initial_length = 0
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

    def build_append(self, time_interval, start_time, end_time, initial_length, is_init, threshold_err,
                     is_retrain, time_retrain, thread_retrain, is_save):
        self.start_time = start_time
        self.time_id = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval
        self.initial_length = initial_length
        self.is_init = is_init
        self.threshold_err = threshold_err
        self.is_retrain = is_retrain
        self.time_retrain = time_retrain
        self.thread_retrain = thread_retrain
        self.is_save = is_save
        for leaf_node in self.rmi[-1]:
            leaf_node.delta_index = Array(self.initial_length)

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
         1. merge delta index into index
         2. update model
         """
        self.logging.info("Update time id: %s" % self.time_id)
        self.logging.info("Insert key time: %s" % (self.insert_time - self.last_insert_time))
        self.logging.info("Insert key io: %s" % (self.insert_io - self.last_insert_io))
        self.last_insert_time = self.insert_time
        self.last_insert_io = self.insert_io
        index_len = 0
        # 1. merge delta index into index
        update_list = [0] * self.stages[-1]
        start_io = self.io_cost
        start_time = time.time()
        for j in range(self.stages[-1]):
            leaf_node = self.rmi[-1][j]
            if leaf_node.delta_index.max_key >= 0:
                update_list[j] = 1
                if leaf_node.index:
                    leaf_node.index.extend(leaf_node.delta_index.index[:leaf_node.delta_index.max_key + 1])
                    leaf_node.index.sort(key=lambda x: x[2])  # 优化：有序数组合并->sorted:2.5->1
                else:
                    leaf_node.index = leaf_node.delta_index.index[:leaf_node.delta_index.max_key + 1]
                leaf_node.delta_index = Array(self.initial_length)
                # IO1: merge data
                self.io_cost += math.ceil(len(leaf_node.index) / ITEMS_PER_PAGE)
            index_len += len(leaf_node.index) + leaf_node.delta_index.size
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
                if update_list[j] == 1:
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
        self.logging.info("Index entry size: %s" % (index_len * ITEM_SIZE))
        self.logging.info("Error bound: %s" % self.model_err())

    def save(self):
        super(DTUSLI, self).save()
        meta_append = np.array((self.start_time, self.time_id, self.time_interval,
                                self.initial_length, self.is_init, self.threshold_err),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'), ("3", 'i4'), ("4", 'i1'), ("5", 'f8')])
        np.save(os.path.join(self.model_path, 'meta_append.npy'), meta_append)
        compute = np.array((self.is_retrain, self.time_retrain, self.thread_retrain, self.is_save),
                           dtype=[("0", 'i1'), ("1", 'i2'), ("2", 'i1'), ("3", 'i1')])
        np.save(os.path.join(self.model_path, 'compute.npy'), compute)

    def load(self):
        super(DTUSLI, self).load()
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

    def size(self):
        """
        structure_size += meta_append.npy
        ie_size
        """
        structure_size, ie_size = super(DTUSLI, self).size()
        structure_size += os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128
        return structure_size, ie_size


def retrain_model(model_path, model_key, inputs, model,
                  weight, cores, train_step, batch_num, learning_rate,
                  is_init, threshold_err, mp_dict):
    inputs = [data[2] for data in inputs]
    old_err = model.max_err - model.min_err
    model.update_error_range(inputs)
    if model.max_err - model.min_err <= threshold_err * old_err:
        mp_dict[model_key] = (model, 0, 0)
    else:
        inputs.insert(0, model.input_min)
        inputs.append(model.input_max)
        inputs_num = len(inputs)
        labels = list(range(0, inputs_num))
        batch_size = 2 ** math.ceil(math.log(inputs_num / batch_num, 2))
        if batch_size < 1:
            batch_size = 1
        tmp_index = NN(model_path, model_key, inputs, labels, True, weight,
                       cores, train_step, batch_size, learning_rate, False, None, None)
        if is_init:
            tmp_index.build_simple(model.matrices if model else None)
        else:
            tmp_index.build_simple(None)
        model.matrices = tmp_index.get_matrices()
        model.min_err = math.floor(tmp_index.min_err)
        model.max_err = math.ceil(tmp_index.max_err)
        mp_dict[model_key] = (model, 1, tmp_index.get_epochs())


def main():
    load_index_from_json = True
    load_index_from_json2 = False
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/dtusli_10w_nyct/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = DTUSLI(model_path=model_path)
    index_name = index.name
    if load_index_from_json:
        super(DTUSLI, index).load()
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
                           initial_length=ITEMS_PER_PAGE,
                           is_init=True,
                           threshold_err=0,
                           is_retrain=True,
                           time_retrain=-1,
                           thread_retrain=3,
                           is_save=False)
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
