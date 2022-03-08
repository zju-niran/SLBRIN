import gc
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import ZOrder, Region, binary_search, Point
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN


class ZMIndex(SpatialIndex):
    def __init__(self, region=Region(-90, 90, -180, 180), model_path=None, train_data_length=None, rmi=None,
                 index_list=None, point_list=None, block_size=None, use_thresholds=None, thresholds=None, stages=None,
                 cores=None, train_steps=None, batch_sizes=None, learning_rates=None, retrain_time_limits=None,
                 thread_pool_size=None):
        super(ZMIndex, self).__init__("ZM Index")
        # nn args
        self.block_size = block_size
        self.use_thresholds = use_thresholds
        self.thresholds = thresholds
        self.stages = stages
        self.stage_length = len(self.stages) if self.stages is not None else 0
        self.cores = cores
        self.train_steps = train_steps
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates
        self.retrain_time_limits = retrain_time_limits
        self.thread_pool_size = thread_pool_size
        self.train_inputs = [[None for i in range(self.stages[i])] for i in range(self.stage_length)]
        self.train_labels = [[None for i in range(self.stages[i])] for i in range(self.stage_length)]

        # zm index args, support predict and query
        self.region = region
        self.model_path = model_path
        self.train_data_length = train_data_length
        self.rmi = [[None for i in range(self.stages[i])] for i in range(self.stage_length)] if rmi is None else rmi
        self.index_list = index_list
        self.point_list = point_list

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
        z_order = ZOrder(dimensions=2, bits=21, region=self.region)
        data["z"] = data.apply(lambda t: z_order.point_to_z(t.x, t.y), 1)
        data.sort_values(by=["z"], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.train_data_length = len(data)
        self.train_inputs[0][0] = data.z.values
        self.train_labels[0][0] = pd.Series(np.arange(0, self.train_data_length) / self.block_size).values
        self.index_list = data.z.values.tolist()
        self.point_list = data[["x", "y"]].values.tolist()

    def build_single_thread(self, curr_stage, current_stage_step, inputs, labels, tmp_dict=None):
        # train model
        i = curr_stage
        j = current_stage_step
        model_path = self.model_path
        model_index = str(i) + "_" + str(j)
        tmp_index = TrainedNN(model_path, model_index, inputs, labels,
                              self.thresholds[i],
                              self.use_thresholds[i],
                              self.cores[i],
                              self.train_steps[i],
                              self.batch_sizes[i],
                              self.learning_rates[i],
                              self.retrain_time_limits[i])
        tmp_index.train()
        tmp_index.plot()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.get_weights(),
                                    self.cores[i],
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

    def build(self, data: pd.DataFrame):
        """
        build index by multi threads
        1. init train z->index data from x/y data
        2. create rmi for train z->index data
        3. clear train data and label to save memory
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
                    pres = self.rmi[i][j].predicts(self.train_inputs[i][j])
                    for ind in range(self.stages[i + 1]):
                        self.train_inputs[i + 1][ind] = self.train_inputs[i][j][np.round(pres) == ind]
                        self.train_labels[i + 1][ind] = self.train_labels[i][j][np.round(pres) == ind]
        # 叶子节点使用线程池训练
        multiprocessing.set_start_method('spawn')  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=self.thread_pool_size)
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

        # 3. clear train data and label to save memory
        self.train_inputs = None
        self.train_labels = None

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
            self.train_data_length = zm_index.train_data_length
            self.rmi = zm_index.rmi
            self.index_list = np.loadtxt(self.model_path + 'index_list.csv', dtype=np.int64, delimiter=",").tolist()
            self.point_list = np.loadtxt(self.model_path + 'point_list.csv', dtype=np.float, delimiter=",").tolist()
            del zm_index

    @staticmethod
    def init_by_dict(d: dict):
        return ZMIndex(region=d['region'],
                       train_data_length=d['train_data_length'],
                       rmi=d['rmi'],
                       index_list=d['index_list'],
                       point_list=d['point_list'])

    def point_query(self, points):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. predict by z and create index scope [pre - min_err, pre + max_err]
        3. binary search in scope
        :param points: list, [x, y]
        :return: list, [pre]
        """
        z_order = ZOrder(dimensions=2, bits=21, region=self.region)
        results = []
        for point in points:
            # 1. compute z from x/y of points
            z_value = z_order.point_to_z(point[0], point[1])
            # 2. predict by z and create index scope [pre - min_err, pre + max_err]
            pre, min_err, max_err = self.predict(z_value)
            left_bound = max((pre - max_err) * self.block_size, 0)
            right_bound = min((pre - min_err) * self.block_size, self.train_data_length - 1)
            # 3. binary search in scope
            result = binary_search(self.index_list, z_value, round(left_bound), round(right_bound))
            if result is not None:
                result /= self.block_size
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
        z_order = ZOrder(dimensions=2, bits=21, region=self.region)
        results = []
        for window in windows:
            # 1. compute z of window_left and window_right
            z_value1 = z_order.point_to_z(window[2], window[0])
            z_value2 = z_order.point_to_z(window[3], window[1])
            # 2. find index_left by point query
            # if point not found, index_left = pre - min_err
            pre1, min_err1, max_err1 = self.predict(z_value1)
            left_bound1 = round(max((pre1 - max_err1) * self.block_size, 0))
            right_bound1 = round(min((pre1 - min_err1) * self.block_size, self.train_data_length - 1))
            index_left = binary_search(self.index_list, z_value1, left_bound1, right_bound1)
            if index_left is None:
                index_left = left_bound1
            # 3. find index_right by point query
            # if point not found, index_right = pre - max_err
            pre2, min_err2, max_err2 = self.predict(z_value2)
            left_bound2 = round(max((pre2 - max_err2) * self.block_size, 0))
            right_bound2 = round(min((pre2 - min_err2) * self.block_size, self.train_data_length - 1))
            index_right = binary_search(self.index_list, z_value2, left_bound2, right_bound2)
            if index_right is None:
                index_right = right_bound2
            # 4. filter all the point of scope[index1, index2] by range(x1/y1/x2/y2).contain(point)
            tmp_results = []
            region = Region(window[0], window[1], window[2], window[3])
            for index in range(index_left, index_right + 1):
                point = self.point_list[index]
                if region.contain_and_border(Point(point[0], point[1])):
                    tmp_results.append(index)
            results.append(tmp_results)
        return results


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
        t = None
        if len(d.keys()) == 8 and d.__contains__("weights") and d.__contains__("core_nums") \
                and d.__contains__("input_min") and d.__contains__("input_max") and d.__contains__("output_min") \
                and d.__contains__("output_max") and d.__contains__("min_err") and d.__contains__("max_err"):
            t = AbstractNN.init_by_dict(d)
        elif len(d.keys()) == 4 and d.__contains__("bottom") and d.__contains__("up") \
                and d.__contains__("left") and d.__contains__("right"):
            t = Region.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "ZM Index":
            t = ZMIndex.init_by_dict(d)
        return t


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_1_filter.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/zm_index_1451w/"
    index = ZMIndex(region=Region(40, 42, -75, -73), model_path=model_path,
                    train_data_length=None, rmi=None, index_list=None,
                    block_size=100,
                    use_thresholds=[False, False],
                    thresholds=[30, 20],
                    stages=[1, 100],
                    cores=[[1, 128, 1], [1, 128, 1]],
                    train_steps=[500, 500],
                    batch_sizes=[1024, 1024],
                    learning_rates=[0.01, 0.01],
                    retrain_time_limits=[40, 20],
                    thread_pool_size=1)
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
    print("*************start point query************")
    point_query_list = train_set_xy.drop("index", axis=1).values.tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    print("Point query time ", search_time)
    print("Not found nums ", pd.Series(results).isna().sum())
    print("*************start range query************")
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    print("Range query time ", search_time)
    range_query_df["query"] = pd.Series(results).apply(len)
    print("Not found nums ", (range_query_df["query"] != range_query_df["count"]).sum())
    print("*************end %s************" % index_name)
