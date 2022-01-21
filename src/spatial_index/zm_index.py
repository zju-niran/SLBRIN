import gc
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('D:/Code/Paper/st-learned-index')
from src.spatial_index.common_utils import ZOrder
from src.spatial_index.spatial_index import SpatialIndex
from src.b_tree import BTree
from src.rmi_keras import TrainedNN, AbstractNN


class ZMIndex(SpatialIndex):
    def __init__(self):
        super(ZMIndex, self).__init__("ZM Index")
        # nn args
        self.block_size = 100
        self.use_thresholds = [True, True]
        self.thresholds = [0.4, 0.5]
        self.stages = [1, 100]
        self.stage_length = len(self.stages)
        self.cores = [[1, 8, 8, 8, 1], [1, 8, 8, 8, 1]]
        self.train_steps = [20000, 20000]
        self.batch_sizes = [5000, 500]
        self.learning_rates = [0.0001, 0.0001]
        self.keep_ratios = [0.9, 0.9]
        self.train_inputs = [[[] for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.train_labels = [[[] for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.train_data_length = 0

        # zm index args
        self.index = [[None for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.normalization_values = [0, 0]  # 查询时用来归一化z
        self.errs = [0, 0]  # 查询时的左右误差边界


    def init_train_data(self, data: pd.DataFrame):
        """
        init train data from x/y data
        1. compute z from data.x and data.y
        2. normalize z by z.min and z.max
        3. sort z and reset index
        4. inputs = z and labels = index / block_size
        :param data: pd.dataframe, [x, y]
        :return: None
        """
        z_order = ZOrder()
        z_values = data.apply(lambda t: z_order.point_to_z(t.x, t.y), 1)
        z_values_min = z_values.min()
        z_values_max = z_values.max()
        self.normalization_values = [z_values_min, z_values_max]
        # z归一化
        z_values_normalization = (z_values - z_values_min) / (z_values_max - z_values_min)
        self.train_data_length = z_values_normalization.size
        self.train_inputs[0][0] = z_values_normalization.sort_values(ascending=True).tolist()
        self.train_labels[0][0] = pd.Series(np.arange(0, self.train_data_length) / self.block_size).tolist()

    def build(self, data: pd.DataFrame):
        """
        build index
        1. init train z->index data from x/y data
        2. create rmi for train z->index data
        """
        # 1. init train z->index data from x/y data
        self.init_train_data(data)
        # 2. create rmi for train z->index data
        # 构建stage_nums结构的树状NNs
        for i in range(0, self.stage_length):
            for j in range(0, self.stages[i]):
                if len(self.train_labels[i][j]) == 0:
                    continue
                inputs = self.train_inputs[i][j]
                labels = []
                # 非叶子结点决定下一层要用的NN是哪个
                if i < self.stage_length - 1:
                    # first stage, calculate how many models in next stage
                    divisor = self.stages[i + 1] * 1.0 / (self.train_data_length / self.block_size)
                    for k in self.train_labels[i][j]:
                        labels.append(int(k * divisor))
                else:
                    labels = self.train_labels[i][j]
                # train model
                model_path = "model_0.0001_5000_adam_drop/" + str(i) + "_" + str(j) + "_weights.best.hdf5"
                print("start train nn in stage: %d, %d" % (i, j))
                tmp_index = TrainedNN(model_path, inputs, labels,
                                      self.thresholds[i],
                                      self.use_thresholds[i],
                                      self.cores[i],
                                      self.train_steps[i],
                                      self.batch_sizes[i],
                                      self.learning_rates[i],
                                      self.keep_ratios[i])
                tmp_index.train()
                # get parameters in model (weight matrix and bias matrix)
                self.index[i][j] = AbstractNN(tmp_index.get_weights(),
                                              self.cores[i],
                                              tmp_index.err,
                                              tmp_index.threshold)
                del tmp_index
                gc.collect()
                if i < self.stage_length - 1:
                    # allocate data into training set for models in next stage
                    pres = self.index[i][j].predict(self.train_inputs[i][j])
                    pres[pres > self.stages[i + 1] - 1] = self.stages[i + 1] - 1
                    for ind in range(len(pres)):
                        self.train_inputs[i + 1][round(pres[ind])].append(self.train_inputs[i][j][ind])
                        self.train_labels[i + 1][round(pres[ind])].append(self.train_labels[i][j][ind])

        # 如果叶节点NN的精度低于threshold，则使用Btree来代替
        for i in range(self.stages[self.stage_length - 1]):
            if self.index[self.stage_length - 1][i] is None:
                continue
            mean_abs_err = self.index[self.stage_length - 1][i].err
            if mean_abs_err > max(self.index[self.stage_length - 1][i].threshold):
                # replace model with BTree if mean error > threshold
                print("Using BTree in leaf model %d with err %f" % (i, mean_abs_err))
                self.index[self.stage_length - 1][i] = BTree(2)
                self.index[self.stage_length - 1][i].build(self.train_inputs[self.stage_length - 1][i],
                                                           self.train_labels[self.stage_length - 1][i])

    def build_single_thread(self, curr_stage, current_stage_step, inputs, labels, tmp_dict=None):
        # train model
        i = curr_stage
        j = current_stage_step
        model_path = "model_0.0001_5000_adam_drop/" + str(i) + "_" + str(j) + "_weights.best.hdf5"
        tmp_index = TrainedNN(model_path, inputs, labels,
                              self.thresholds[i],
                              self.use_thresholds[i],
                              self.cores[i],
                              self.train_steps[i],
                              self.batch_sizes[i],
                              self.learning_rates[i],
                              self.keep_ratios[i])
        tmp_index.train()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.get_weights(),
                                    self.cores[i])
        del tmp_index
        gc.collect()
        if tmp_dict is not None:
            tmp_dict[j] = abstract_index
        else:
            self.index[i][j] = abstract_index

    def build_multi_thread(self, data: pd.DataFrame, thread_pool_size=3):
        """
        build index by multi threads
        1. init train z->index data from x/y data
        2. create rmi for train z->index data
        3. compute err border by train_y - rmi.predict(train_x)
        4. clear train data and label to save memory
        """
        # 1. init train z->index data from x/y data
        self.init_train_data(data)
        # 2. create rmi for train z->index data
        # 构建stage_nums结构的树状NNs
        for i in range(0, self.stage_length - 1):
            for j in range(0, self.stages[i]):
                if len(self.train_labels[i][j]) == 0:
                    continue
                inputs = self.train_inputs[i][j]
                labels = []
                # 非叶子结点决定下一层要用的NN是哪个
                # first stage, calculate how many models in next stage
                divisor = self.stages[i + 1] * 1.0 / (self.train_data_length / self.block_size)
                for k in self.train_labels[i][j]:
                    labels.append(int(k * divisor))
                # train model
                self.build_single_thread(i, j, inputs, labels)
                # allocate data into training set for models in next stage
                pres = self.index[i][j].predict(self.train_inputs[i][j])
                pres[pres > self.stages[i + 1] - 1] = self.stages[i + 1] - 1
                for ind in range(len(pres)):
                    self.train_inputs[i + 1][round(pres[ind])].append(self.train_inputs[i][j][ind])
                    self.train_labels[i + 1][round(pres[ind])].append(self.train_labels[i][j][ind])
        # 叶子节点使用线程池训练
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        i = self.stage_length - 1
        task_size = self.stages[i]
        for j in range(task_size):
            inputs = self.train_inputs[i][j]
            labels = self.train_labels[i][j]
            pool.apply_async(self.build_single_thread, (i, j, inputs, labels, mp_dict))
        pool.close()
        pool.join()
        # 如果叶节点NN的精度低于threshold，则使用Btree来代替
        for i in range(self.stages[self.stage_length - 1]):
            if self.index[self.stage_length - 1][i] is None:
                continue
            mean_abs_err = self.index[self.stage_length - 1][i].err
            if mean_abs_err > max(self.index[self.stage_length - 1][i].threshold):
                # replace model with BTree if mean error > threshold
                print("Using BTree in leaf model %d with err %f" % (i, mean_abs_err))
                self.index[self.stage_length - 1][i] = BTree(2)
                self.index[self.stage_length - 1][i].build(self.train_inputs[self.stage_length - 1][i],
                                                           self.train_labels[self.stage_length - 1][i])
        for (key, value) in mp_dict.items():
            self.index[i][key] = value

        # 3. compute err border by train_y - rmi.predict(train_x)
        self.get_err()

        # 4. clear train data and label to save memory
        self.train_inputs = None
        self.train_labels = None

    def predict(self, key):
        """
        predict index from key
        1. predict the index by rmi and return pre
        :param key: float
        :return: the index predicted by rmi
        """
        leaf_model = 0
        for i in range(0, self.stage_length - 1):
            leaf_model = round(self.index[i][leaf_model].predict(key)[0])
            if leaf_model > self.stages[i + 1] - 1:
                leaf_model = self.stages[i + 1] - 1
        pre = self.index[i][leaf_model].predict(key)[0]
        return pre

        """
        save zm index into json file
        :return: None
        """
        with open(self.model_file, "w") as f:
            json.dump(self.index, f, default=lambda o: o.__dict__, ensure_ascii=False)

    def load(self):
        """
        load zm index from json file
        :return: None
        """
        with open(self.model_file, "r") as f:
            self.index = json.load(f, object_hook=AbstractNN.init_by_dict)

    def get_err(self):
        """
        get rmi err = rmi.predict(train_x) - train_y
        :return:
        """
        min_err, max_err = 0, 0
        for i in range(self.train_data_length):
            if i == 7949:
                print(i)
            pre = self.predict(self.train_inputs[0][0][i])
            err = pre - self.train_labels[0][0][i]
            if err < 0:
                if err < min_err:
                    min_err = err
            else:
                if err > max_err:
                    max_err = err
        self.errs = [abs(min_err), max_err]

    def point_query(self, data: pd.DataFrame):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. normalize z by z.min and z.max
        3. predict by z and create index scope [pre - min_err, pre + max_err]
        4. binary search in scope
        :param data: pd.DataFrame, [x, y]
        :return: pd.DataFrame, [pre, min_err, max_err]
        """
        z_order = ZOrder()
        z_values = data.apply(lambda t: z_order.point_to_z(t.x, t.y), 1)
        # z归一化
        z_values_normalization = (z_values - self.normalization_values[0]) / (
                self.normalization_values[1] - self.normalization_values[0])

        pres = z_values_normalization.apply(self.predict)
        scope = data.iloc[(pres - self.errs[0]) * self.block_size:(pres + self.errs[1]) * self.block_size]
        value = self.binary_search(scope, data.index * self.block_size)
        return value

    # def range_query(self, data: pd.DataFrame):
    #     """
    #     query index by x1/y1/x2/y2 range
    #     1. compute z1, z2 from x1/y1/x2/y2 of data
    #     2. normalize z1 and z2 by z.min and z.max
    #     3. point query from z1 and z2 to index_z1 and index_z2
    #     4. filter all the point of scope[index_z1, index_z2] by range(x1/y1/x2/y2).contain(point)
    #     :param data: pd.DataFrame, [x1, y1, x2, y2]
    #     :return: pd.DataFrame, [pre, min_err, max_err]
    #     """
    #     z_order = ZOrder()
    #     z_value = z_order.point_to_z(df[lng_col][i], df[lat_col][i])
    #     # z归一化
    #     min_z_value = min(z_values)
    #     max_z_value = max(z_values)
    #     for i in range(df.count()[0]):
    #         z_value_normalization = (z_values[i] - min_z_value) / (max_z_value - min_z_value)
    #         z_values_normalization.append(z_value_normalization)

    def binary_search(self, nums, x):
        """
        nums: Sorted array from smallest to largest
        x: Target number
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == x:
                return mid
            if nums[mid] < x:
                left = mid + 1
            else:
                right = mid - 1
        return None


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_2_100000_random.csv'
    # read_data_and_search(path, index, None, None, 7, 8)
    z_col, index_col = 7, 8
    train_set_xy = pd.read_csv(path, header=None, usecols=[2, 3], names=["x", "y"])
    test_ratio = 0.5  # 测试集占总数据集的比例
    test_set_xy = train_set_xy.sample(n=int(len(train_set_xy) * test_ratio), random_state=1)
    # create index
    start_time = time.time()
    index = ZMIndex()
    index_name = index.name
    print("*************start %s************" % index_name)
    print("Start Build")
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        index.build_multi_thread(train_set_xy)
        index.save()
    end_time = time.time()
    build_time = end_time - start_time
    print("Build %s time " % index_name, build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    err = index.point_query(test_set_xy)
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_xy)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_xy)
    print("mean error = ", mean_error)
    print("*************end %s************" % index_name)
