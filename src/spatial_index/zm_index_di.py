import logging
import math
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/SBRIN')
from src.experiment.common_utils import load_data, Distribution, data_region, data_precision, load_query
from src.spatial_index.zm_index import AbstractNN
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
        # 更新所需：
        self.start_time = None
        self.cur_time_interval = None
        self.time_interval = None

    def build_append(self, time_interval, start_time, end_time):
        self.start_time = start_time
        self.cur_time_interval = math.ceil((end_time - start_time) / time_interval)
        self.time_interval = time_interval

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
         1. merge delta index into index
         2. update model
         """
        leaf_nodes = self.rmi[-1]
        retrain_model_num = 0
        retrain_model_epoch = 0
        for j in range(0, self.stages[-1]):
            leaf_node = leaf_nodes[j]
            if leaf_node.delta_index:
                # 1. merge delta index into index
                if leaf_node.index:
                    leaf_node.index.extend(leaf_node.delta_index)
                    leaf_node.index.sort(key=lambda x: x[2])  # 优化：有序数组合并->sorted:2.5->1
                else:
                    leaf_node.index = leaf_node.delta_index
                leaf_node.delta_index = []
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
        self.logging.info("Retrain model num: %s" % retrain_model_num)
        self.logging.info("Retrain model epoch: %s" % retrain_model_epoch)

    def save(self):
        super(ZMIndexDeltaInsert, self).save()
        meta_append = np.array((self.start_time, self.cur_time_interval, self.time_interval),
                               dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4')])
        np.save(os.path.join(self.model_path, 'meta_append.npy'), meta_append)

    def load(self):
        super(ZMIndexDeltaInsert, self).load()
        meta_append = np.load(os.path.join(self.model_path, 'meta_append.npy'), allow_pickle=True).item()
        self.start_time = meta_append[0]
        self.cur_time_interval = meta_append[1]
        self.time_interval = meta_append[2]

    def size(self):
        structure_size, ie_size = super(ZMIndexDeltaInsert, self).size()
        structure_size += os.path.getsize(os.path.join(self.model_path, "meta_append.npy")) - 128
        return structure_size, ie_size


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/zmdi_10w_nyct/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = ZMIndexDeltaInsert(model_path=model_path)
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
                           end_time=1359676799)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    logging.info("Model precision avg: %s" % index.model_err())
    # point_query_list = load_query(data_distribution, 0).tolist()
    # start_time = time.time()
    # results = index.point_query(point_query_list)
    # end_time = time.time()
    # search_time = (end_time - start_time) / len(point_query_list)
    # logging.info("Point query time: %s" % search_time)
    # np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))
    point_query_list = load_query(data_distribution, 0).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
