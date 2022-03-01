import gc
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('D:/Code/Paper/st-learned-index')
from src.brin import BRIN, RegularPage, RevMapPage, MetaPage
from src.spatial_index.quad_tree import QuadTree
from src.spatial_index.common_utils import ZOrder, Region
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN


class GeoHashModelIndex(SpatialIndex):
    def __init__(self, region=Region(-90, 90, -180, 180), max_num=10000, model_path=None, train_data_length=None,
                 brin=None, gm_dict=None, index_list=None):
        super(GeoHashModelIndex, self).__init__("GeoHash Model Index")
        # nn args
        self.block_size = 100
        self.use_threshold = True
        self.threshold = 2
        self.core = [1, 128, 1]
        self.train_step = 30000
        self.batch_size = 1024
        self.learning_rate = 0.01
        self.keep_ratio = 0.9
        self.retrain_time_limit = 20
        self.thread_pool_size = 3

        # geohash model index args, support predict and query
        self.region = region
        self.max_num = max_num
        self.model_path = model_path
        self.train_data_length = train_data_length
        self.brin = brin
        self.gm_dict = gm_dict if gm_dict is not None else {}
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
        self.train_data_length = len(data)
        data["z_index"] = pd.Series(np.arange(0, self.train_data_length) / self.block_size)
        self.index_list = pd.DataFrame({'key': data.z,
                                        "key_index": data.z_index})

    def build(self, data: pd.DataFrame):
        """
        build index
        1. init train z->index data from x/y data
        2. split data by quad tree: geohash->data_list
        3. create brin index
        4. create zm-model(stage=1) for every leaf node
        5. clear train data and label to save memory
        """
        # 1. init train z->index data from x/y data
        self.init_train_data(data)
        # 2. split data by quad tree
        quad_tree = QuadTree(region=self.region, max_num=self.max_num)
        quad_tree.build(data, z=True)
        quad_tree.geohash()
        split_data = quad_tree.geohash_items_map
        # 3. create brin index
        self.brin = BRIN(version=0, pages_per_range=None, revmap_page_maxitems=500, regular_page_maxitems=500)
        self.brin.build_by_quad_tree(quad_tree)
        # 4. in every part data, create zm-model
        multiprocessing.set_start_method('spawn')  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=self.thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        for geohash_key in split_data:
            points = split_data[geohash_key]["items"]
            inputs = np.array([item.z for item in points])
            labels = np.array([item.index for item in points])
            if len(labels) == 0:
                continue
            pool.apply_async(self.build_single_thread, (1, geohash_key, inputs, labels, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.gm_dict[key] = value
        # 5. clear train data and label to save memory

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
                              self.keep_ratio,
                              self.retrain_time_limit)
        tmp_index.train()
        tmp_index.plot()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.get_weights(),
                                    self.core,
                                    tmp_index.train_x_min,
                                    tmp_index.train_x_max,
                                    tmp_index.train_y_min,
                                    tmp_index.train_y_max,
                                    tmp_index.min_err,
                                    tmp_index.max_err)
        del tmp_index
        gc.collect()
        tmp_dict[j] = abstract_index

    def save(self):
        """
        save gm index into json file
        :return: None
        """
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        self.index_list.to_csv(self.model_path + 'index_list.csv', sep=',', header=True, index=False)
        with open(self.model_path + 'gm_index.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)

    def load(self):
        """
        load gm index from json file
        :return: None
        """
        with open(self.model_path + 'gm_index.json', "r") as f:
            gm_index = json.load(f, cls=MyDecoder)
            self.train_data_length = gm_index.train_data_length
            self.brin = gm_index.brin
            self.gm_dict = gm_index.gm_dict
            self.index_list = pd.read_csv(self.model_path + 'index_list.csv',
                                          float_precision='round_trip')  # round_trip保留小数位数
            del gm_index

    @staticmethod
    def init_by_dict(d: dict):
        return GeoHashModelIndex(region=d['region'],
                                 max_num=d['max_num'],
                                 train_data_length=d['train_data_length'],
                                 brin=d['brin'],
                                 gm_dict=d['gm_dict'],
                                 index_list=d['index_list'])

    def point_query(self, data: pd.DataFrame):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. normalize z by z.min and z.max
        3. predict the leaf model by brin
        4. predict by leaf model and create index scope [pre - min_err, pre + max_err]
        5. binary search in scope
        :param data: pd.DataFrame, [x, y]
        :return: pd.DataFrame, [pre]
        """
        z_order = ZOrder()
        results = []
        # 1. compute z from x/y of points
        # 2. normalize z by z.min and z.max
        z_values = data.apply(lambda t: z_order.point_to_z(t.x, t.y, self.region) / z_order.max_z, 1)
        # 3. predicted the leaf model by brin
        leaf_model_indexes = self.brin.point_query(z_values)
        for i in range(len(z_values)):
            z = z_values[i]
            leaf_model_index = leaf_model_indexes[i]
            leaf_model = self.gm_dict[leaf_model_index]
            # 4. predict by z and create index scope [pre - min_err, pre + max_err]
            pre, min_err, max_err = leaf_model.predict(z)[0], leaf_model.min_err, leaf_model.max_err
            left_bound = max((pre - max_err) * self.block_size, 0)
            right_bound = min((pre - min_err) * self.block_size, self.train_data_length - 1)
            # 4. binary search in scope
            result = self.binary_search(self.index_list, z, int(round(left_bound)), int(round(right_bound)))
            results.append(result)
        return pd.Series(results)

    # TODO: 无法处理有重复的数组
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
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Region):
            return obj.__dict__
        elif isinstance(obj, GeoHashModelIndex):
            return obj.__dict__
        elif isinstance(obj, AbstractNN):
            return obj.__dict__
        elif isinstance(obj, BRIN):
            return obj.__dict__
        elif isinstance(obj, MetaPage):
            return obj.__dict__
        elif isinstance(obj, RevMapPage):
            return obj.__dict__
        elif isinstance(obj, RegularPage):
            return obj.__dict__
        else:
            return super(MyEncoder, self).default(obj)


class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)

    def dict_to_object(self, d):
        if len(d.keys()) == 8 and d.__contains__("weights") and d.__contains__("core_nums") \
                and d.__contains__("input_min") and d.__contains__("input_max") and d.__contains__("output_min") \
                and d.__contains__("output_max") and d.__contains__("min_err") and d.__contains__("max_err"):
            t = AbstractNN.init_by_dict(d)
        elif len(d.keys()) == 4 and d.__contains__("bottom") and d.__contains__("up") \
                and d.__contains__("left") and d.__contains__("right"):
            t = Region.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "GeoHash Model Index":
            t = GeoHashModelIndex.init_by_dict(d)
        elif len(d.keys()) == 3 and d.__contains__("version") \
                and d.__contains__("pages_per_range") and d.__contains__("last_revmap_page"):
            t = MetaPage.init_by_dict(d)
        elif len(d.keys()) == 2 and d.__contains__("id") and d.__contains__("pages"):
            t = RevMapPage.init_by_dict(d)
        elif len(d.keys()) == 8 and d.__contains__("id") and d.__contains__("itemoffsets") and d.__contains__(
                "blknums") and d.__contains__("attnums") and d.__contains__("allnulls") and d.__contains__(
            "hasnulls") and d.__contains__("placeholders") and d.__contains__("values"):
            t = RegularPage.init_by_dict(d)
        elif len(d.keys()) == 7 and d.__contains__("version") and d.__contains__("pages_per_range") \
                and d.__contains__("revmap_page_maxitems") and d.__contains__(
            "regular_page_maxitems") and d.__contains__("meta_page") and d.__contains__(
            "revmap_pages") and d.__contains__("regular_pages"):
            t = BRIN.init_by_dict(d)
        else:
            t = d
        return t


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_2_100000_random.csv'
    # read_data_and_search(path, index, None, None, 7, 8)
    z_col, index_col = 7, 8
    train_set_xy = pd.read_csv(path, header=None, usecols=[2, 3], names=["x", "y"])
    # create index
    model_path = "model/gm_index_2022-03-01/"
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
