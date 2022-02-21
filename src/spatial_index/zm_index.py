import gc
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('D:/Code/Paper/st-learned-index')
from src.spatial_index.common_utils import ZOrder, Region
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN


class ZMIndex(SpatialIndex):
    def __init__(self, region=Region(-90, 90, -180, 180), model_path=None, train_data_length=None, rmi=None, errs=None,
                 index_list=None):
        super(ZMIndex, self).__init__("ZM Index")
        # nn args
        self.block_size = 100
        self.use_thresholds = [True, True]  # 是否使用thresholds来提前结束训练
        self.thresholds = [30, 20]
        self.stages = [1, 100]
        self.stage_length = len(self.stages)
        self.cores = [[1, 8, 8, 8, 1], [1, 8, 8, 8, 1]]
        self.train_steps = [20000, 20000]
        self.batch_sizes = [5000, 500]
        self.learning_rates = [0.0001, 0.0001]
        self.keep_ratios = [0.9, 0.9]
        self.train_inputs = [[None for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.train_labels = [[None for i in range(self.stages[i])] for i in range(self.stage_length)]

        # zm index args, support predict and query
        self.region = region
        self.model_path = model_path
        self.train_data_length = train_data_length
        self.rmi = [[None for i in range(self.stages[i])] for i in range(self.stage_length)] if rmi is None else rmi
        self.errs = errs  # 查询时的左右误差边界
        self.index_list = index_list  # 索引列

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
        z_values = data.apply(lambda t: z_order.point_to_z(t.x, t.y, self.region), 1)
        # z归一化
        z_values_normalization = z_values / z_order.max_z
        self.train_data_length = z_values_normalization.size
        self.train_inputs[0][0] = z_values_normalization.sort_values(ascending=True).values
        self.train_labels[0][0] = pd.Series(np.arange(0, self.train_data_length) / self.block_size).values

    def build_single_thread(self, curr_stage, current_stage_step, inputs, labels, tmp_dict=None):
        # train model
        i = curr_stage
        j = current_stage_step
        model_path = self.model_path + "models/" + str(i) + "_" + str(j) + "_weights.best.hdf5"
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
                                    self.cores[i],
                                    tmp_index.train_x_min,
                                    tmp_index.train_x_max,
                                    tmp_index.train_y_min,
                                    tmp_index.train_y_max)
        del tmp_index
        gc.collect()
        if tmp_dict is not None:
            tmp_dict[j] = abstract_index
        else:
            self.rmi[i][j] = abstract_index

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
                if self.train_labels[i][j] is None:
                    continue
                else:
                    inputs = self.train_inputs[i][j]
                    labels = []
                    # 非叶子结点决定下一层要用的NN是哪个
                    # first stage, calculate how many models in next stage
                    divisor = self.stages[i + 1] * 1.0 / (self.train_data_length / self.block_size)
                    labels = (self.train_labels[i][j] * divisor).astype(int)
                    # train model
                    self.build_single_thread(i, j, inputs, labels)
                    # allocate data into training set for models in next stage
                    pres = self.rmi[i][j].predict(self.train_inputs[i][j])
                    for ind in range(self.stages[i + 1]):
                        self.train_inputs[i + 1][ind] = self.train_inputs[i][j][np.round(pres) == ind]
                        self.train_labels[i + 1][ind] = self.train_labels[i][j][np.round(pres) == ind]
        # 叶子节点使用线程池训练
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        i = self.stage_length - 1
        task_size = self.stages[i]
        for j in range(task_size):
            inputs = self.train_inputs[i][j]
            labels = self.train_labels[i][j]
            if labels is None or len(labels) == 0:
                continue
            pool.apply_async(self.build_single_thread, (i, j, inputs, labels, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.rmi[i][key] = value

        # 3. compute err border by train_y - rmi.predict(train_x)
        self.errs = self.get_err()

        # 4. clear train data and label to save memory
        self.index_list = pd.DataFrame({'key': self.train_inputs[0][0], "key_index": self.train_labels[0][0]})
        self.train_inputs = None
        self.train_labels = None

    def predict(self, key):
        """
        predict index from key
        1. predict the leaf_model by rmi
        2. predict the index by leaf_model
        :param key: float
        :return: the index predicted by rmi
        """
        # 1. predict the leaf_model by rmi
        leaf_model = 0
        for i in range(0, self.stage_length - 1):
            leaf_model = round(self.rmi[i][leaf_model].predict(key)[0])
            if leaf_model > self.stages[i + 1] - 1:
                leaf_model = self.stages[i + 1] - 1
        # 2. predict the index by leaf_model
        pre = self.rmi[self.stage_length - 1][leaf_model].predict(key)[0]
        return pre

    def save(self):
        """
        save zm index into json file
        :return: None
        """
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        self.index_list.to_csv(self.model_path + 'index_list.csv', sep=',', header=True, index=False)
        with open(self.model_path + 'zm_index.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)

    def load(self):
        """
        load zm index from json file
        :return: None
        """
        with open(self.model_path + 'zm_index.json', "r") as f:
            zm_index = json.load(f, cls=MyDecoder)
            self.train_data_length = zm_index.train_data_length
            self.rmi = zm_index.rmi
            self.errs = zm_index.errs
            self.index_list = pd.read_csv(self.model_path + 'index_list.csv',
                                          float_precision='round_trip')  # round_trip保留小数位数
            del zm_index

    @staticmethod
    def init_by_dict(d: dict):
        return ZMIndex(region=d['region'],
                       train_data_length=d['train_data_length'],
                       rmi=d['rmi'],
                       errs=d['errs'],
                       index_list=d['index_list'])

    def get_err(self):
        """
        get rmi err = rmi.predict(train_x) - train_y
        :return: [min_err, max_err]
        """
        min_err, max_err = 0, 0
        for i in range(self.train_data_length):
            pre = self.predict(self.train_inputs[0][0][i])
            err = self.train_labels[0][0][i] - pre
            if err < 0:
                if err < min_err:
                    min_err = err
            else:
                if err > max_err:
                    max_err = err
        return [abs(min_err), max_err]

    def point_query(self, data: pd.DataFrame):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. normalize z by z.min and z.max
        3. predict by z and create index scope [pre - min_err, pre + max_err]
        4. binary search in scope
        :param data: pd.DataFrame, [x, y]
        :return: pd.DataFrame, [pre]
        """
        z_order = ZOrder()
        # 写法1：list
        results = []
        for index, point in data.iterrows():
            z_value = z_order.point_to_z(point.x, point.y, self.region)
            # z归一化
            z = z_value / z_order.max_z
            pre = self.predict(z)
            left_bound = max((pre - self.errs[0]) * self.block_size, 0)
            right_bound = min((pre + self.errs[1]) * self.block_size, self.train_data_length)
            result = self.binary_search(self.index_list, z, int(round(left_bound)), int(round(right_bound)))
            results.append(result)
        return pd.Series(results)
        # 写法2：pd.DataFrame
        # z_values = data.apply(lambda t: z_order.point_to_z(t.x, t.y, self.region), 1)
        # # z归一化
        # data["z"] = (z_values - self.z_values_normalization_min_max[0]) / (
        #         self.z_values_normalization_min_max[1] - self.z_values_normalization_min_max[0])
        # data["pres"] = data.z.apply(self.predict)
        # data["left_bound"] = data.pres.apply(lambda pre: max((pre - self.errs[0]) * self.block_size, 0))
        # data["right_bound"] = data.pres.apply(
        #     lambda pre: min((pre + self.errs[1]) * self.block_size, self.train_data_length))
        # results = data.apply(
        #     lambda t: self.binary_search(self.index_list, t.z, int(round(t.left_bound)), int(round(t.right_bound))), 1)
        # return results

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

    def binary_search(self, nums, x, left, right):
        """
        binary search x in nums[left, right]
        :param nums: pd.DataFrame, [key, key_index], index table
        :param x: key
        :param left:
        :param right:
        :return: key index
        """
        while left <= right:
            mid = (left + right) // 2
            if nums.iloc[mid].key == x:
                return nums.iloc[mid].key_index
            if nums.iloc[mid].key < x:
                left = mid + 1
            else:
                right = mid - 1
        return None


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return None
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, Region):
            return obj.__dict__
        elif isinstance(obj, ZMIndex):
            return obj.__dict__
        elif isinstance(obj, AbstractNN):
            return obj.__dict__
        else:
            return super(MyEncoder, self).default(obj)


class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)

    def dict_to_object(self, d):
        if len(d.keys()) == 2 and d.__contains__("weights") and d.__contains__("core_nums"):
            t = AbstractNN.init_by_dict(d)
        elif len(d.keys()) == 4 and d.__contains__("bottom") and d.__contains__("up") \
                and d.__contains__("left") and d.__contains__("right"):
            t = Region.init_by_dict(d)
        else:
            t = ZMIndex.init_by_dict(d)
        return t


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_2_100000_random.csv'
    # read_data_and_search(path, index, None, None, 7, 8)
    z_col, index_col = 7, 8
    train_set_xy = pd.read_csv(path, header=None, usecols=[2, 3], names=["x", "y"])
    # create index
    model_path = "model/zm_index_2022-02-04/"
    index = ZMIndex(region=Region(40, 42, -75, -73), model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build_multi_thread(train_set_xy, thread_pool_size=4)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        index.save()
    start_time = time.time()
    result = index.point_query(train_set_xy)
    end_time = time.time()
    search_time = (end_time - start_time) / len(train_set_xy)
    print("Search time ", search_time)
    print("Not found nums ", result.isna().sum())
    print("*************end %s************" % index_name)
