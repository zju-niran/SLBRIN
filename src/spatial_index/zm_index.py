import gc
import logging
import math
import multiprocessing
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region, biased_search, normalize_input_minmax, denormalize_output_minmax, \
    sigmoid
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex
from src.learned_model import TrainedNN
from src.learned_model_simple import TrainedNN as TrainedNN_Simple


class ZMIndex(SpatialIndex):
    def __init__(self, model_path=None, geohash=None, train_data_length=None, stage_length=0, rmi=None):
        super(ZMIndex, self).__init__("ZM Index")
        self.geohash = geohash
        self.train_data_length = train_data_length
        self.stage_length = stage_length
        self.rmi = rmi
        self.geohash_index = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def build(self, data_list, data_precision, region, use_thresholds, thresholds, stages, cores, train_steps,
              batch_nums, learning_rates, retrain_time_limits, thread_pool_size, save_nn, weight):
        """
        build index
        1. ordering x/y point by geohash
        2. create rmi to train geohash->key data
        """
        self.geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        self.stage_length = len(stages)
        train_inputs = [[[] for i in range(stages[i])] for i in range(self.stage_length)]
        train_labels = [[[] for i in range(stages[i])] for i in range(self.stage_length)]
        self.rmi = [[None for i in range(stages[i])] for i in range(self.stage_length)]
        # 1. ordering x/y point by geohash
        self.geohash_index = data_list
        self.train_data_length = len(self.geohash_index) - 1
        train_inputs[0][0] = [data[2] for data in data_list]
        train_labels[0][0] = list(range(0, self.train_data_length + 1))
        # 2. create rmi to train geohash->key data
        # 构建stage_nums结构的树状NNs
        for i in range(self.stage_length - 1):
            for j in range(stages[i]):
                if train_labels[i][j] is None:
                    continue
                else:
                    inputs = train_inputs[i][j]
                    # 非叶子结点决定下一层要用的NN是哪个
                    # first stage, calculate how many models in next stage
                    divisor = stages[i + 1] * 1.0 / (self.train_data_length + 1)
                    labels = [int(k * divisor) for k in train_labels[i][j]]
                    # train model
                    self.build_nn(i, j, inputs, labels, use_thresholds[i], thresholds[i], cores[i],
                                  train_steps[i], batch_nums[i], learning_rates[i], retrain_time_limits[i],
                                  save_nn, weight)
                    # allocate data into training set for models in next stage
                    for ind in range(len(train_inputs[i][j])):
                        # pick model in next stage with output of this model
                        pre = round(self.rmi[i][j].predict(train_inputs[i][j][ind]))
                        train_inputs[i + 1][pre].append(train_inputs[i][j][ind])
                        train_labels[i + 1][pre].append(train_labels[i][j][ind])
        # 叶子节点使用线程池训练
        multiprocessing.set_start_method('spawn', force=True)
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()
        i = self.stage_length - 1
        task_size = stages[i]
        for j in range(task_size):
            inputs = train_inputs[i][j]
            labels = train_labels[i][j]
            if labels is None or len(labels) == 0:
                continue
            pool.apply_async(self.build_nn,
                             (i, j, inputs, labels, use_thresholds[i], thresholds[i], cores[i], train_steps[i],
                              batch_nums[i], learning_rates[i], retrain_time_limits[i], save_nn, weight, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.rmi[i][key] = value

    def build_nn(self, curr_stage, current_stage_step, inputs, labels, use_threshold, threshold, core,
                 train_step, batch_num, learning_rate, retrain_time_limit, save_nn, weight, tmp_dict=None):
        batch_size = 2 ** math.ceil(math.log(len(inputs) / batch_num, 2))
        if batch_size < 1:
            batch_size = 1
        i = curr_stage
        j = current_stage_step
        model_key = str(i) + "_" + str(j)
        if save_nn is False:
            tmp_index = TrainedNN_Simple(self.model_path, model_key, inputs, labels, core, train_step, batch_size,
                                         learning_rate, weight)
        else:
            tmp_index = TrainedNN(self.model_path, model_key, inputs, labels, use_threshold, threshold, core,
                                  train_step, batch_size, learning_rate, retrain_time_limit, weight)
        tmp_index.train()
        abstract_index = AbstractNN(tmp_index.get_weights(),
                                    core,
                                    tmp_index.train_x_min,
                                    tmp_index.train_x_max,
                                    tmp_index.train_y_min,
                                    tmp_index.train_y_max,
                                    math.floor(tmp_index.min_err),
                                    math.ceil(tmp_index.max_err))
        del tmp_index
        gc.collect()
        if tmp_dict:
            tmp_dict[j] = abstract_index
        else:
            self.rmi[i][j] = abstract_index

    def predict(self, key):
        """
        predict key from key
        1. predict the leaf_model by rmi
        2. return the less max key when leaf model is None
        3. predict the key by leaf_model
        :param key: float
        :return: the key predicted by rmi, min_err and max_err of leaf_model
        """
        # 1. predict the leaf_model by rmi
        leaf_model_key = 0
        for i in range(0, self.stage_length - 1):
            leaf_model_key = round(self.rmi[i][leaf_model_key].predict(key))
        # 2. return the less max key when leaf model is None
        if self.rmi[-1][leaf_model_key] is None:
            while self.rmi[-1][leaf_model_key] is None:
                if leaf_model_key < 0:
                    return 0, 0, 0
                else:
                    leaf_model_key -= 1
            return self.rmi[-1][leaf_model_key].output_max, 0, 0
        # 3. predict the key by leaf_model
        leaf_model = self.rmi[-1][leaf_model_key]
        pre = leaf_model.predict(key)
        return round(pre), leaf_model.min_err, leaf_model.max_err

    def save(self):
        zmin_meta = np.array((self.geohash.data_precision,
                              self.geohash.region.bottom, self.geohash.region.up,
                              self.geohash.region.left, self.geohash.region.right,
                              self.stage_length, self.train_data_length),
                             dtype=[("0", 'i4'),
                                    ("1", 'f8'), ("2", 'f8'), ("3", 'f8'), ("4", 'f8'),
                                    ("5", 'i4'), ("6", 'i4')])
        np.save(os.path.join(self.model_path, 'zmin_meta.npy'), zmin_meta)
        np.save(os.path.join(self.model_path, 'zmin_rmi.npy'), self.rmi)
        np.save(os.path.join(self.model_path, 'geohash_index.npy'), self.geohash_index)

    def load(self):
        zmin_rmi = np.load(self.model_path + 'zmin_rmi.npy', allow_pickle=True)
        zmin_meta = np.load(self.model_path + 'zmin_meta.npy', allow_pickle=True).item()
        geohash_index = np.load(self.model_path + 'geohash_index.npy', allow_pickle=True)
        region = Region(zmin_meta[1], zmin_meta[2], zmin_meta[3], zmin_meta[4])
        self.geohash = Geohash.init_by_precision(data_precision=zmin_meta[0], region=region)
        self.stage_length = zmin_meta[5]
        self.train_data_length = zmin_meta[6]
        self.rmi = zmin_rmi.tolist()
        self.geohash_index = geohash_index

    def size(self):
        """
        size = zmin_rmi.npy + zmin_meta.npy + geohash_index.npy
        """
        # 实际上：os.path.getsize(os.path.join(self.model_path, "zmin_meta.npy")) - 128 - 64
        # 理论上：只存geohash_length/stage_length/train_data_length三个int即可
        return os.path.getsize(os.path.join(self.model_path, "zmin_rmi.npy")) - 128 + \
               4 * 3 + \
               os.path.getsize(os.path.join(self.model_path, "geohash_index.npy")) - 128

    def point_query_single(self, point):
        """
        1. compute geohash from x/y of point
        2. predict by geohash and create key scope [pre - min_err, pre + max_err]
        3. binary search in scope
        """
        # 1. compute geohash from x/y of point
        gh = self.geohash.encode(point[0], point[1])
        # 2. predict by geohash and create key scope [pre - min_err, pre + max_err]
        pre, min_err, max_err = self.predict(gh)
        l_bound = max(pre - max_err, 0)
        r_bound = min(pre - min_err, self.train_data_length)
        # 3. binary search in scope
        geohash_keys = biased_search(self.geohash_index, 2, gh, pre, l_bound, r_bound)
        return [self.geohash_index[key][3] for key in geohash_keys]

    def range_query_single(self, window):
        """
        1. compute geohash from window_left and window_right
        2. find key_left by point query
        3. find key_right by point query
        4. filter all the points of scope[key_left, key_right] by range(x1/y1/x2/y2).contain(point)
        """
        region = Region(window[0], window[1], window[2], window[3])
        # 1. compute z of window_left and window_right
        gh1 = self.geohash.encode(window[2], window[0])
        gh2 = self.geohash.encode(window[3], window[1])
        # 2. find key_left by point query
        # if point not found, key_left = pre - min_err
        pre1, min_err1, max_err1 = self.predict(gh1)
        l_bound1 = max(pre1 - max_err1, 0)
        r_bound1 = min(pre1 - min_err1, self.train_data_length)
        key_left = biased_search(self.geohash_index, 2, gh1, pre1, l_bound1, r_bound1)
        key_left = l_bound1 if len(key_left) == 0 else min(key_left)
        # 3. find key_right by point query
        # if point not found, key_right = pre - max_err
        pre2, min_err2, max_err2 = self.predict(gh2)
        l_bound2 = max(pre2 - max_err2, 0)
        r_bound2 = min(pre2 - min_err2, self.train_data_length)
        key_right = biased_search(self.geohash_index, 2, gh2, pre2, l_bound2, r_bound2)
        key_right = r_bound2 if len(key_right) == 0 else max(key_right)
        # 4. filter all the point of scope[key1, key2] by range(x1/y1/x2/y2).contain(point)
        return [self.geohash_index[key][3] for key in range(key_left, key_right + 1)
                if region.contain_and_border_by_list(self.geohash_index[key])]


class AbstractNN:
    def __init__(self, weights, core_nums, input_min, input_max, output_min, output_max, min_err, max_err):
        self.weights = weights
        self.core_nums = core_nums
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max
        self.min_err = min_err
        self.max_err = max_err

    # @memoize
    # model.predict有小偏差，可能是exp的e和elu的e不一致
    def predict(self, input_key):
        """
        单个key的矩阵计算
        """
        y = normalize_input_minmax(input_key, self.input_min, self.input_max)
        for i in range(len(self.core_nums) - 2):
            y = sigmoid(y * self.weights[i * 2] + self.weights[i * 2 + 1])
        y = y * self.weights[-2] + self.weights[-1]
        return denormalize_output_minmax(y[0, 0], self.output_min, self.output_max)


# @profile(precision=8)
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # data_path = '../../data/table/trip_data_1_filter_10w.npy'
    data_path = '../../data/index/trip_data_1_filter_10w_sorted.npy'
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
        # data_list = np.load(data_path, allow_pickle=True)[:, [10, 11, -1]]
        data_list = np.load(data_path, allow_pickle=True)
        # 按照pagesize=4096, prefetch=256, size(pointer)=4, size(x/y/g)=8, meta单独一个page, rmi(2375大小)每个模型1个page
        # 因为精确过滤是否需要xy判断，因此geohash索引相当于存储x/y/g/key四个值=8+8+8+4=28
        # 10w数据，[1, 100]参数下：
        # 单次扫描IO为读取meta(和rmi连续存所以忽略)+读取每个stage的rmi+读取叶stage对应geohash数据=2+1
        # 索引体积为geohash索引+rmi=28*10w+4096*(1+100)
        index.build(data_list=data_list,
                    data_precision=6,
                    region=Region(40, 42, -75, -73),
                    use_thresholds=[False, False],
                    thresholds=[5, 20],
                    stages=[1, 100],
                    cores=[[1, 128, 1], [1, 128, 1]],
                    train_steps=[5000, 5000],
                    batch_nums=[64, 64],
                    learning_rates=[0.1, 0.1],
                    retrain_time_limits=[4, 2],
                    thread_pool_size=6,
                    save_nn=True,
                    weight=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    logging.info("Index size: %s" % index.size())
    path = '../../data/query/point_query_10w.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/range_query_10w.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time:  %s" % search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    # path = '../../data/query/knn_query_10w.npy'
    # knn_query_list = np.load(path, allow_pickle=True).tolist()
    # start_time = time.time()
    # results = index.knn_query(knn_query_list)
    # end_time = time.time()
    # search_time = (end_time - start_time) / len(knn_query_list)
    # logging.info("KNN query time:  %s" % search_time)
    # np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    # path = '../../data/table/trip_data_2_filter_10w.npy'
    # insert_data_list = np.load(path, allow_pickle=True)[:, [10, 11, -1]]
    # start_time = time.time()
    # index.insert(insert_data_list)
    # end_time = time.time()
    # logging.info("Insert time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
