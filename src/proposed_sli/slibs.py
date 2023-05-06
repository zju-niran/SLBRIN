import logging
import multiprocessing
import os
import time

import numpy as np

from src.experiment.common_utils import load_data, Distribution, load_query, data_precision, data_region
from src.mlp import MLP
from src.mlp_simple import MLPSimple
from src.sli.zm_index import ZMIndex, Node, build_nn, Array
from src.utils.geohash_utils import Geohash

PAGE_SIZE = 4096
MODEL_SIZE = 2000
ITEM_SIZE = 8 * 3 + 4  # 28
MODELS_PER_PAGE = int(PAGE_SIZE / MODEL_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class SLIBS(ZMIndex):
    """
    空间学习型索引基础框架（Basic Structure for Spatial Learned Index，SLIBS）
    1. 基本思路：先降维+后RMI
    2. 其他改进：在训练集中采用最大最小值归一化方法和分区边界控制方法，保证分段CDF在段间的单调性
    2.1. 分区边界提取过程见get_leaf_bound
    2.2. 分区边界加入过程见p115~123
    """

    def __init__(self, model_path=None):
        super(SLIBS, self).__init__(model_path)

    def build(self, data_list, is_sorted, data_precision, region, is_new, is_simple, weight,
              stages, cores, train_steps, batch_nums, learning_rates, use_thresholds, thresholds, retrain_time_limits,
              thread_pool_size):
        """
        different from zm_index
        1. add the key bound into the inputs of each leaf node to reduce the err bound
        """
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
                # 1. add the key bound into the inputs of each leaf node
                key_left_bounds = self.get_leaf_bound()
                for j in range(task_size):
                    inputs = [data[2] for data in train_input[j]]
                    left_bound = key_left_bounds[j]
                    right_bound = key_left_bounds[j + 1] if j + 1 < task_size else 1 << self.geohash.sum_bits
                    if inputs and not (left_bound <= inputs[0] and inputs[-1] < right_bound):
                        raise RuntimeError("the inputs [%f, %f] of leaf node %d exceed the limits [%f, %f]" % (
                            inputs[0], inputs[-1], j, left_bound, right_bound))
                    inputs.insert(0, key_left_bounds[j])
                    inputs.append(key_left_bounds[j + 1] if j + 1 < task_size else 1 << self.geohash.sum_bits)
                    labels = list(range(0, len(inputs)))
                    pool.apply_async(build_nn, (self.model_path, i, j, inputs, labels, is_new, is_simple,
                                                weight, core, train_step, batch_num, learning_rate,
                                                use_threshold, threshold, retrain_time_limit, mp_list))
                pool.close()
                pool.join()
                nodes = [Node(train_input[j], mp_list[j], Array()) for j in range(task_size)]
                for node in nodes:
                    node.model.output_max -= 2  # remove the key bound
            self.rmi[i] = nodes
            # clear the data already used
            train_inputs[i] = None
            train_labels[i] = None

    def get_leaf_bound(self):
        """
        get the key bound of leaf node by rmi
        1. use non-leaf rmi as func
        2. get the key bound for per leaf node
        e.g. rmi(x-1)=i-1, rmi(x)=i, rmi(y-1)=i, rmi(y)=i+1, so key_bound_i=(x, y),
        """
        leaf_node_len = self.stages[-1]
        key_left_bounds = [0] * leaf_node_len
        max_key = 1 << self.geohash.sum_bits
        for i in range(1, leaf_node_len):
            left = key_left_bounds[i - 1]
            right = max_key
            while left <= right:
                mid = (left + right) >> 1
                if self.get_leaf_node(mid) >= i:
                    if self.get_leaf_node(mid - 1) >= i:
                        right = mid - 1
                    else:
                        key_left_bounds[i] = mid
                        break
                else:
                    left = mid + 1
            if left > right:
                key_left_bounds[i] = left
        return key_left_bounds


class NN(MLP):
    def __init__(self, model_path, model_key, train_x, train_y, is_new, weight, core, train_step, batch_size,
                 learning_rate, use_threshold, threshold, retrain_time_limit):
        self.name = "SLIBS NN"
        # train_x的是有序的，归一化不需要计算最大最小值
        train_x_min = train_x[0]
        train_x_max = train_x[-1]
        train_x = (np.array(train_x) - train_x_min) / (train_x_max - train_x_min) - 0.5
        train_y_min = train_y[0]
        train_y_max = train_y[-1]
        train_y = (np.array(train_y) - train_y_min) / (train_y_max - train_y_min)
        super().__init__(model_path, model_key, train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         is_new, weight, core, train_step, batch_size, learning_rate, use_threshold, threshold,
                         retrain_time_limit)


class NNSimple(MLPSimple):
    def __init__(self, train_x, train_y, weight, core, train_step, batch_size, learning_rate):
        self.name = "SLIBS NN"
        # train_x的是有序的，归一化不需要计算最大最小值
        train_x_min = train_x[0]
        train_x_max = train_x[-1]
        train_x = (np.array(train_x) - train_x_min) / (train_x_max - train_x_min) - 0.5
        train_y_min = train_y[0]
        train_y_max = train_y[-1]
        train_y = (np.array(train_y) - train_y_min) / (train_y_max - train_y_min)
        super().__init__(train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                         weight, core, train_step, batch_size, learning_rate)


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/slibs_10w/"
    data_distribution = Distribution.NYCT_10W_SORTED
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = SLIBS(model_path=model_path)
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
                    is_new=True,
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
