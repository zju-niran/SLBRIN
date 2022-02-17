import gc
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

from src.spatial_index.quad_tree import QuadTree

sys.path.append('D:/Code/Paper/st-learned-index')
from src.spatial_index.common_utils import ZOrder, Region
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN


class GeoHashModelIndex(SpatialIndex):
    def __init__(self, region=Region(-90, 90, -180, 180), max_num=10000, model_path=None, train_data_length=None,
                 rmi=None, errs=None, index_list=None):
        super(GeoHashModelIndex, self).__init__("GeoHash Model Index")
        # nn args
        self.block_size = 100
        self.use_threshold = True
        self.threshold = 5
        self.core = [1, 8, 8, 8, 1]
        self.train_step = 20000
        self.batch_size = 500
        self.learning_rate = 0.0001
        self.keep_ratio = 0.9

        # geohash model index args, support predict and query
        self.region = region
        self.max_num = max_num  # 单个geohash内的数据量
        self.model_path = model_path
        self.train_data_length = train_data_length
        self.rmi = None
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
        data["z"] = z_values / z_order.max_z
        data.sort_values(by=["z"], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.train_data_length = data.size
        data["z_index"] = pd.Series(np.arange(0, self.train_data_length) / self.block_size)
        self.index_list = pd.DataFrame({'key': data.z,
                                        "key_index": data.z_index})

    def build(self, data: pd.DataFrame):
        """
        build index
        1. init train z->index data from x/y data
        2. split data by quad tree: geohash->data_list
        3. create zm-model(stage=1) for every leaf node
        """
        # 1. init train z->index data from x/y data
        self.init_train_data(data)
        # 2. split data by quad tree
        quadtree = QuadTree(region=self.region, max_num=self.max_num)
        quadtree.build(data, z=True)
        quadtree.geohash()
        split_data = quadtree.geohash_items_map
        # 2. in every part data, create zm-model
        pool = multiprocessing.Pool(processes=5)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        for geohash_key in split_data:
            points = split_data[geohash_key]["items"]
            inputs = [item.z for item in points]
            labels = [item.index for item in points]
            if len(labels) == 0:
                continue
            pool.apply_async(self.build_single_thread, (1, geohash_key, inputs, labels, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.rmi[0][key] = value

        # 3. compute err border by train_y - rmi.predict(train_x)
        self.errs = self.get_err()

        # 4. clear train data and label to save memory
        self.index_list = pd.DataFrame({'key': self.train_inputs[0][0], "key_index": self.train_labels[0][0]})
        self.train_inputs = None
        self.train_labels = None

    def build_single_thread(self, curr_stage, current_stage_step, inputs, labels, tmp_dict=None):
        # train model
        i = curr_stage
        j = current_stage_step
        model_path = self.model_path + "models/" + str(i) + "_" + str(j) + "_weights.best.hdf5"
        tmp_index = TrainedNN(model_path, inputs, labels,
                              self.threshold,
                              self.use_threshold,
                              self.core,
                              self.train_step,
                              self.batch_size,
                              self.learning_rate,
                              self.keep_ratio)
        tmp_index.train()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.get_weights(),
                                    self.cores[i])
        del tmp_index
        gc.collect()
        if tmp_dict is not None:
            tmp_dict[j] = abstract_index
        else:
            self.rmi[i][j] = abstract_index

    def predict(self, data: pd.DataFrame):
        """
        predict index from z data
        1. predict by z and return [pre, min_err, max_err]
        :param data: pd.dataframe, [x, y]
        :return: pd.dataframe, [pre, min_err, max_err]
        """
        leaf_model = 0
        for i in range(0, self.stage_length - 1):
            leaf_model = self.index[i][leaf_model].predict(data.z)
        pre = self.index[i][leaf_model].predict(data.z)
        [min_err, max_err] = self.index[i][leaf_model].mean_err
        return [pre, min_err, max_err]

    def point_query(self, points: pd.DataFrame):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. normalize z by z.min and z.max
        3. predict by z and create index scope [pre - min_err, pre + max_err]
        4. binary search in scope
        :param data: pd.dataframe, [x, y]
        :return: pd.dataframe, [pre, min_err, max_err]
        """
        z_order = ZOrder()
        z_values = points.apply(lambda t: z_order.point_to_z(t.x, t.y, self.region), 1)
        # z归一化
        z_values_normalization = z_values / z_order.max_z
        data = pd.DataFrame({'z': z_values_normalization})
        [pre, min_err, max_err] = self.predict(data.z)
        scope = points.iloc[(pre - err[0]) * self.block_size:(pre + err[1]) * self.block_size]
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
    # create index
    model_path = "model/gm_index_2022-02-17/"
    index = GeoHashModelIndex(region=Region(40, 42, -75, -73), max_num=1000, model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(train_set_xy)
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
