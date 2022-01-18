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
        self.index = [[None for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.train_data_length = 0
        self.normalization_values = [0, 0]
        self.train_inputs = [[[] for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.train_labels = [[[] for i in range(self.stages[i])] for i in range(self.stage_length)]

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
                model_path = "model_0.0001_5000_adam_drop/" + str(i) + "_" + str(j) + "/"
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


    def predict(self, points):
        stage_length = len(self.stages)
        leaf_model = 0
        for i in range(0, stage_length - 1):
            leaf_model = self.index[i][leaf_model].predict(points.z)
        pre = self.index[i][leaf_model].predict(points.z)
        err = self.index[i][leaf_model].mean_err
        scope = list(range((pre - err) * self.block_size, (pre + err) * self.block_size))
        value = self.binary_search(scope, points.index * self.block_size)
        return value

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
    test_set_point = train_set_xy.sample(n=int(len(train_set_xy) * test_ratio), random_state=1)
    # create index
    start_time = time.time()
    index = ZMIndex()
    index_name = index.name
    print("*************start %s************" % index_name)
    print("Start Build")
    index.build(train_set_point)
    load_index_from_json = False
    end_time = time.time()
    build_time = end_time - start_time
    print("Build %s time " % index_name, build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    for ind in range(len(test_set_point)):
        err += index.predict(test_set_point[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_point)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_point)
    print("mean error = ", mean_error)
    print("*************end %s************" % index_name)
