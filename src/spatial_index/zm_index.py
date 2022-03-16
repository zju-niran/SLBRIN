import gc
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region, biased_search, ZOrder
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN


class ZMIndex(SpatialIndex):
    def __init__(self, model_path=None, z_order=None, train_data_length=None, stage_length=0, rmi=None, index_list=None,
                 point_list=None):
        super(ZMIndex, self).__init__("ZM Index")
        self.model_path = model_path
        self.z_order = z_order
        self.train_data_length = train_data_length
        self.stage_length = stage_length
        self.rmi = rmi
        self.index_list = index_list
        self.point_list = point_list

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
        self.train_data_length = len(data) - 1
        self.index_list = data.z.tolist()
        self.point_list = data[["x", "y"]].values.tolist()

    def build(self, data: pd.DataFrame, data_precision, region, use_thresholds, thresholds, stages, cores, train_steps,
              batch_sizes, learning_rates, retrain_time_limits, thread_pool_size):
        """
        build index by multi threads
        1. init train z->index data from x/y data
        2. create rmi for train z->index data
        """
        self.z_order = ZOrder(data_precision=data_precision, region=region)
        self.stage_length = len(stages)
        train_inputs = [[[] for i in range(stages[i])] for i in range(self.stage_length)]
        train_labels = [[[] for i in range(stages[i])] for i in range(self.stage_length)]
        self.rmi = [[None for i in range(stages[i])] for i in range(self.stage_length)]
        # 1. init train z->index data from x/y data
        self.init_train_data(data)
        train_inputs[0][0] = self.index_list
        train_labels[0][0] = np.arange(0, self.train_data_length + 1).tolist()
        # 2. create rmi for train z->index data
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
                    self.build_single_thread(i, j, inputs, labels, use_thresholds[i], thresholds[i], cores[i],
                                             train_steps[i], batch_sizes[i], learning_rates[i], retrain_time_limits[i])
                    # allocate data into training set for models in next stage
                    for ind in range(len(train_inputs[i][j])):
                        # pick model in next stage with output of this model
                        pre = round(self.rmi[i][j].predict(train_inputs[i][j][ind]))
                        train_inputs[i + 1][pre].append(train_inputs[i][j][ind])
                        train_labels[i + 1][pre].append(train_labels[i][j][ind])
        # 叶子节点使用线程池训练
        multiprocessing.set_start_method('spawn')  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        i = self.stage_length - 1
        task_size = stages[i]
        for j in range(task_size):
            inputs = train_inputs[i][j]
            labels = train_labels[i][j]
            if labels is None or len(labels) == 0:
                continue
            pool.apply_async(self.build_single_thread,
                             (i, j, inputs, labels, use_thresholds[i], thresholds[i], cores[i],
                              train_steps[i], batch_sizes[i], learning_rates[i], retrain_time_limits[i], mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.rmi[i][key] = value

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
        if tmp_dict is not None:
            tmp_dict[j] = abstract_index
        else:
            self.rmi[i][j] = abstract_index

    def predict(self, key):
        """
        predict index from key
        1. predict the leaf_model by rmi
        2. predict the index by leaf_model
        :param key: float
        :return: the index predicted by rmi, min_err and max_err of leaf_model
        """
        # 1. predict the leaf_model by rmi
        leaf_model_index = 0
        for i in range(0, self.stage_length - 1):
            leaf_model_index = round(self.rmi[i][leaf_model_index].predict(key))
        # 2. predict the index by leaf_model
        leaf_model = self.rmi[self.stage_length - 1][leaf_model_index]
        pre = leaf_model.predict(key)
        return pre, leaf_model.min_err, leaf_model.max_err

    def save(self):
        """
        save zm index into json file
        :return: None
        """
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        np.savetxt(self.model_path + 'index_list.csv', self.index_list, delimiter=',', fmt='%d')
        np.savetxt(self.model_path + 'point_list.csv', self.point_list, delimiter=',', fmt='%f,%f')
        with open(self.model_path + 'zm_index.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)

    def load(self):
        """
        load zm index from json file
        :return: None
        """
        with open(self.model_path + 'zm_index.json', "r") as f:
            zm_index = json.load(f, cls=MyDecoder)
            self.z_order = zm_index.z_order
            self.train_data_length = zm_index.train_data_length
            self.stage_length = zm_index.stage_length
            self.rmi = zm_index.rmi
            self.index_list = np.loadtxt(self.model_path + 'index_list.csv', dtype=np.int64, delimiter=",").tolist()
            self.point_list = np.loadtxt(self.model_path + 'point_list.csv', dtype=float, delimiter=",").tolist()
            del zm_index

    @staticmethod
    def init_by_dict(d: dict):
        return ZMIndex(z_order=d['z_order'],
                       train_data_length=d['train_data_length'],
                       stage_length=d['stage_length'],
                       rmi=d['rmi'])

    def save_to_dict(self):
        return {
            'name': self.name,
            'z_order': self.z_order,
            'train_data_length': self.train_data_length,
            'stage_length': self.stage_length,
            'rmi': self.rmi
        }

    def point_query(self, points):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. predict by z and create index scope [pre - min_err, pre + max_err]
        3. binary search in scope
        :param points: list, [x, y]
        :return: list, [pre]
        """
        results = []
        for point in points:
            # 1. compute z from x/y of points
            z_value = self.z_order.point_to_z(point[0], point[1])
            # 2. predict by z and create index scope [pre - min_err, pre + max_err]
            pre, min_err, max_err = self.predict(z_value)
            pre_init = int(pre)  # int比round快一倍
            left_bound = max(round(pre - max_err), 0)
            right_bound = min(round(pre - min_err), self.train_data_length)
            # 3. binary search in scope
            result = biased_search(self.index_list, z_value, pre_init, left_bound, right_bound)
            results.append(result)
        return results

    def range_query(self, windows):
        """
        query index by x1/y1/x2/y2 window
        1. compute z from window_left and window_right
        2. find index_left by point query
        3. find index_right by point query
        4. filter all the points of scope[index_left, index_right] by range(x1/y1/x2/y2).contain(point)
        :param windows: list, [x1, y1, x2, y2]
        :return: list, [pres]
        """
        results = []
        for window in windows:
            # 1. compute z of window_left and window_right
            z_value1 = self.z_order.point_to_z(window[2], window[0])
            z_value2 = self.z_order.point_to_z(window[3], window[1])
            # 2. find index_left by point query
            # if point not found, index_left = pre - min_err
            pre1, min_err1, max_err1 = self.predict(z_value1)
            pre1_init = int(pre1)
            left_bound1 = max(round(pre1 - max_err1), 0)
            right_bound1 = min(round(pre1 - min_err1), self.train_data_length)
            index_left = biased_search(self.index_list, z_value1, pre1_init, left_bound1, right_bound1)
            index_left = left_bound1 if len(index_left) == 0 else min(index_left)
            # 3. find index_right by point query
            # if point not found, index_right = pre - max_err
            pre2, min_err2, max_err2 = self.predict(z_value2)
            pre2_init = int(pre2)
            left_bound2 = max(round(pre2 - max_err2), 0)
            right_bound2 = min(round(pre2 - min_err2), self.train_data_length)
            index_right = biased_search(self.index_list, z_value2, pre2_init, left_bound2, right_bound2)
            index_right = right_bound2 if len(index_right) == 0 else max(index_right)
            # 4. filter all the point of scope[index1, index2] by range(x1/y1/x2/y2).contain(point)
            tmp_results = []
            region = Region(window[0], window[1], window[2], window[3])
            for index in range(index_left, index_right + 1):
                point = self.point_list[index]
                if region.contain_and_border(point[0], point[1]):
                    tmp_results.append(index)
            results.append(tmp_results)
        return results


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Region):
            return obj.__dict__
        elif isinstance(obj, ZOrder):
            return obj.save_to_dict()
        elif isinstance(obj, ZMIndex):
            return obj.save_to_dict()
        elif isinstance(obj, AbstractNN):
            return obj.__dict__
        else:
            return super(MyEncoder, self).default(obj)


class MyDecoder(json.JSONDecoder):
    def __init__(self):
        json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)

    def dict_to_object(self, d):
        t = None
        if len(d.keys()) == 8 and d.__contains__("weights") and d.__contains__("core_nums") \
                and d.__contains__("input_min") and d.__contains__("input_max") and d.__contains__("output_min") \
                and d.__contains__("output_max") and d.__contains__("min_err") and d.__contains__("max_err"):
            t = AbstractNN.init_by_dict(d)
        elif len(d.keys()) == 4 and d.__contains__("bottom") and d.__contains__("up") \
                and d.__contains__("left") and d.__contains__("right"):
            t = Region.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "Z Order":
            t = ZOrder.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "ZM Index":
            t = ZMIndex.init_by_dict(d)
        else:
            t = d
        return t


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_1_filter.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/zm_index_1451w/"
    index = ZMIndex(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(data=train_set_xy, data_precision=6, region=Region(40, 42, -75, -73),
                    use_thresholds=[False, False],
                    thresholds=[30, 20],
                    stages=[1, 100],
                    cores=[[1, 128, 1], [1, 128, 1]],
                    train_steps=[500, 500],
                    batch_sizes=[1024, 1024],
                    learning_rates=[0.01, 0.01],
                    retrain_time_limits=[40, 20],
                    thread_pool_size=1)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        index.save()
    path = '../../data/trip_data_1_point_query.csv'
    point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
    point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    print("Point query time ", search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    print("Range query time ", search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    print("*************end %s************" % index_name)
