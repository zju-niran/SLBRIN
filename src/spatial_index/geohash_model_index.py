import gc
import json
import multiprocessing
import os
import sys
import time

import line_profiler
import numpy as np
import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')
from src.brin import BRIN, RegularPage, RevMapPage, MetaPage
from src.spatial_index.quad_tree import QuadTree
from src.spatial_index.common_utils import ZOrder, Region, biased_search
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN


class GeoHashModelIndex(SpatialIndex):
    def __init__(self, model_path=None, z_order=None, train_data_length=None, brin=None, gm_dict=None,
                 index_list=None):
        super(GeoHashModelIndex, self).__init__("GeoHash Model Index")
        self.model_path = model_path
        self.z_order = z_order
        self.train_data_length = train_data_length
        self.brin = brin
        self.gm_dict = gm_dict
        self.index_list = index_list

    def init_train_data(self, data: pd.DataFrame):
        """
        init train data from x/y data
        1. compute z from data.x and data.y
        2. inputs = z and labels = range(0, data_length)
        :param data: pd.dataframe, [x, y]
        :return: None
        """
        data["z"] = data.apply(lambda t: self.z_order.point_to_z(t.x, t.y), 1)
        data.sort_values(by=["z"], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.train_data_length = len(data)
        self.index_list = data.z.tolist()

    def build(self, data: pd.DataFrame, max_num, data_precision, region, use_threshold, threshold, core, train_step,
              batch_size, learning_rate, retrain_time_limit, thread_pool_size):
        """
        build index
        1. init train z->index data from x/y data
        2. split data by quad tree: geohash->data_list
        3. create brin index
        4. create zm-model(stage=1) for every leaf node
        """
        self.z_order = ZOrder(data_precision=data_precision, region=region)
        self.gm_dict = {}
        # 1. init train z->index data from x/y data
        self.init_train_data(data)
        # 2. split data by quad tree
        quad_tree = QuadTree(region=region, max_num=max_num, data_precision=data_precision)
        quad_tree.build(data, z=True)
        quad_tree.geohash(self.z_order)
        split_data = quad_tree.geohash_items_map
        # 3. create brin index
        self.brin = BRIN(version=0, pages_per_range=None, revmap_page_maxitems=500, regular_page_maxitems=500)
        self.brin.build_by_quad_tree(quad_tree)
        # 4. in every part data, create zm-model
        multiprocessing.set_start_method('spawn')  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        for geohash_key in split_data:
            points = split_data[geohash_key]["items"]
            inputs = []
            labels = []
            for point in points:
                inputs.append(point.z)
                labels.append(point.index)
            if len(labels) == 0:
                continue
            pool.apply_async(self.build_single_thread, (1, geohash_key, inputs, labels, use_threshold, threshold, core,
                                                        train_step, batch_size, learning_rate, retrain_time_limit,
                                                        mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.gm_dict[key] = value

    def build_single_thread(self, curr_stage, current_stage_step, inputs, labels, use_threshold, threshold,
                            core, train_step, batch_size, learning_rate, retrain_time_limit, tmp_dict=None):
        # train model
        i = curr_stage
        j = current_stage_step
        model_index = str(i) + "_" + str(j)
        tmp_index = TrainedNN(self.model_path, model_index, inputs, labels,
                              use_threshold,
                              threshold,
                              core,
                              train_step,
                              batch_size,
                              learning_rate,
                              retrain_time_limit)
        tmp_index.train()
        tmp_index.plot()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.get_weights(),
                                    core,
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
        np.savetxt(self.model_path + 'index_list.csv', self.index_list, delimiter=',', fmt='%d')
        self.index_list = None
        with open(self.model_path + 'gm_index.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)

    def load(self):
        """
        load gm index from json file
        :return: None
        """
        with open(self.model_path + 'gm_index.json', "r") as f:
            gm_index = json.load(f, cls=MyDecoder)
            self.z_order = gm_index.z_order
            self.train_data_length = gm_index.train_data_length
            self.brin = gm_index.brin
            self.gm_dict = gm_index.gm_dict
            self.index_list = np.loadtxt(self.model_path + 'index_list.csv', delimiter=",").tolist()
            del gm_index

    @staticmethod
    def init_by_dict(d: dict):
        return GeoHashModelIndex(z_order=d['z_order'],
                                 train_data_length=d['train_data_length'],
                                 brin=d['brin'],
                                 gm_dict=d['gm_dict'])

    def point_query(self, points):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. predict the leaf model by brin
        3. predict by leaf model and create index scope [pre - min_err, pre + max_err]
        4. binary search in scope
        :param points: list, [x, y]
        :return: list, [pre]
        """
        results = []
        for point in points:
            # 1. compute z from x/y of points
            z = self.z_order.point_to_z(point[0], point[1])
            # 2. predicted the leaf model by brin
            leaf_model_index = self.brin.point_query(z)
            if leaf_model_index is None:
                result = None
            else:
                leaf_model = self.gm_dict[leaf_model_index]
                # 3. predict by z and create index scope [pre - min_err, pre + max_err]
                pre, min_err, max_err = leaf_model.predict(z), leaf_model.min_err, leaf_model.max_err
                pre_init = int(pre)  # int比round快一倍
                left_bound = max(round(pre - max_err), 0)
                right_bound = min(round(pre - min_err), self.train_data_length - 1)
                # 4. binary search in scope
                result = biased_search(self.index_list, z, pre_init, left_bound, right_bound)
            results.append(result)
        return results

    def range_query(self, data: pd.DataFrame):
        """
        query index by x1/y1/x2/y2 window
        1. get
        2. normalize z by z.min and z.max
        3. predict the leaf model by brin
        4. predict by leaf model and create index scope [pre - min_err, pre + max_err]
        5. binary search in scope
        :param data: pd.DataFrame, [x, y]
        :return: pd.DataFrame, [pre]
        """


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
        elif isinstance(obj, ZOrder):
            return obj.dict()
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
        elif len(d.keys()) == 2 and d.__contains__("data_precision") and d.__contains__("region"):
            t = ZOrder.init_by_dict(d)
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
    path = '../../data/trip_data_1_filter.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/gm_index_1451w/"
    index = GeoHashModelIndex(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(data=train_set_xy, max_num=1000, data_precision=6, region=Region(40, 42, -75, -73),
                    use_threshold=False,
                    threshold=2,
                    core=[1, 128, 1],
                    train_step=500,
                    batch_size=1024,
                    learning_rate=0.01,
                    retrain_time_limit=20,
                    thread_pool_size=1)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        index.save()
    train_set_xy_list = np.delete(train_set_xy.values, 0, 1).tolist()
    start_time = time.time()
    result = index.point_query(train_set_xy_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(train_set_xy_list)
    print("Search time ", search_time)
    print("Not found nums ", pd.Series(result).isna().sum())
    print("*************end %s************" % index_name)
