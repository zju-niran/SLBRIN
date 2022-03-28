import gc
import json
import logging
import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')
from src.sbrin import SBRIN
from src.spatial_index.common_utils import Region, biased_search, Point, biased_search_almost
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN
from src.rmi_keras_simple import TrainedNN as TrainedNN_Simple


class GeoHashModelIndex(SpatialIndex):
    def __init__(self, model_path=None, geohash=None, sbrin=None, gm_dict=None, index_list=None, point_list=None):
        super(GeoHashModelIndex, self).__init__("GeoHash Model Index")
        self.model_path = model_path
        self.geohash = geohash
        self.sbrin = sbrin
        self.gm_dict = gm_dict
        self.index_list = index_list
        self.point_list = point_list

    def init_train_data(self, data: pd.DataFrame, load_data):
        """
        init train data from x/y data
        1. compute z from data.x and data.y
        2. inputs = z and labels = range(0, data_length)
        """
        if load_data:
            self.index_list = np.loadtxt(self.model_path + 'index_list.csv', delimiter=",").tolist()
            self.point_list = np.loadtxt(self.model_path + 'point_list.csv', dtype=float, delimiter=",").tolist()
        else:
            data["z"] = data.apply(lambda t: self.geohash.point_to_z(t.x, t.y), 1)
            data.sort_values(by=["z"], ascending=True, inplace=True)
            data.reset_index(drop=True, inplace=True)
            self.index_list = data.z.tolist()
            self.point_list = data[["x", "y"]].values.tolist()

    def build(self, data: pd.DataFrame, threshold_number, data_precision, region, use_threshold, threshold, core,
              train_step,
              batch_size, learning_rate, retrain_time_limit, thread_pool_size, load_data, save_nn):
        """
        build index
        1. init train z->index data from x/y data
        2. create brin index
        3. create zm-model(stage=1) for every leaf node
        """
        self.geohash = Geohash.init_by_precision(data_precision=data_precision, region=region)
        # 1. init train z->index data from x/y data
        self.init_train_data(data, load_data)
        # 2. create brin index
        self.sbrin = SBRIN()
        self.sbrin.build(self.index_list, self.geohash.sum_bits, region, threshold_number, data_precision)
        # 3. in every part data, create zm-model
        multiprocessing.set_start_method('spawn', force=True)  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()  # 使用共享dict暂存index[i]的所有model
        block_num = self.sbrin.size + 1
        self.gm_dict = [None for i in range(block_num)]
        for index in range(block_num):
            z_index_bound = self.sbrin.blkindexes[index]
            if self.sbrin.blknums == 0:
                continue
            inputs = self.index_list[z_index_bound[0]:z_index_bound[1] + 1]
            labels = list(range(z_index_bound[0], z_index_bound[1] + 1))
            pool.apply_async(self.build_single_thread, (1, index, inputs, labels, use_threshold, threshold, core,
                                                        train_step, batch_size, learning_rate, retrain_time_limit,
                                                        save_nn, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.gm_dict[key] = value

    def build_single_thread(self, curr_stage, current_stage_step, inputs, labels, use_threshold, threshold,
                            core, train_step, batch_size, learning_rate, retrain_time_limit, save_nn, tmp_dict=None):
        # train model
        i = curr_stage
        j = current_stage_step
        if save_nn is False:
            logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                                level=logging.INFO,
                                format="%(asctime)s - %(levelname)s - %(message)s",
                                datefmt="%m/%d/%Y %H:%M:%S %p")
            start_time = time.time()
            tmp_index = TrainedNN_Simple(inputs, labels, core, train_step, batch_size, learning_rate)
            tmp_index.train()
            end_time = time.time()
            logging.info("Model index: %s, Train time: %s" % (j, end_time - start_time))
        else:
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
        np.savetxt(self.model_path + 'point_list.csv', self.point_list, delimiter=',', fmt='%f,%f')
        with open(self.model_path + 'gm_index.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)

    def load(self):
        """
        load gm index from json file
        :return: None
        """
        with open(self.model_path + 'gm_index.json', "r") as f:
            gm_index = json.load(f, cls=MyDecoder)
            self.geohash = gm_index.geohash
            self.sbrin = gm_index.sbrin
            self.gm_dict = gm_index.gm_dict
            self.index_list = np.loadtxt(self.model_path + 'index_list.csv', delimiter=",").tolist()
            self.point_list = np.loadtxt(self.model_path + 'point_list.csv', dtype=float, delimiter=",").tolist()
            del gm_index

    @staticmethod
    def init_by_dict(d: dict):
        return GeoHashModelIndex(geohash=d['geohash'],
                                 sbrin=d['sbrin'],
                                 gm_dict=d['gm_dict'])

    def save_to_dict(self):
        return {
            'name': self.name,
            'geohash': self.geohash,
            'sbrin': self.sbrin,
            'gm_dict': self.gm_dict
        }

    def point_query_single(self, point):
        """
        query index by x/y point
        1. compute z from x/y of points
        2. predict the leaf model by sbrin
        3. predict by leaf model and create index scope [pre - min_err, pre + max_err]
        4. binary search in scope
        """
        # 1. compute z from x/y of point
        z = self.geohash.point_to_z(point[0], point[1])
        # 2. predicted the leaf model by sbrin
        lm_index = self.sbrin.point_query(z)
        lm_indexes = self.sbrin.blkindexes[lm_index]
        lm = self.gm_dict[lm_index]
        if lm is None:
            return None
        else:
            # 3. predict by z and create index scope [pre - min_err, pre + max_err]
            pre, min_err, max_err = lm.predict(z), lm.min_err, lm.max_err
            # 4. binary search in scope
            # 优化: round->int:2->1
            return biased_search(self.index_list, z, int(pre),
                                 max(round(pre - max_err), lm_indexes[0]),
                                 min(round(pre - min_err), lm_indexes[1]))

    def range_query_single_old(self, window):
        """
        query index by x1/y1/x2/y2 window
        1. compute z from window_left and window_right
        2. get block1 and block2 by sbrin,
        and validate the relation in spatial between query window and blocks[block1, block2]
        3. for different relation, use different method to handle the points
        3.1 if window contain the block, add all the items into results
        3.2 if window intersect or within the block
        3.2.1 get the min_z/max_z of intersect part
        3.2.2 get the min_index/max_index by nn predict and biased search
        3.2.3 filter all the point of scope[min_index/max_index] by range.contain(point)
        主要耗时间：两次z的predict和最后的精确过滤，0.1, 0.1 , 0.6
        # TODO: 由于build sbrin的时候region移动了，导致这里的查询不准确了
        """
        region = Region(window[0], window[1], window[2], window[3])
        # 1. compute z of window_left and window_right
        z_value1 = self.geohash.point_to_z(window[2], window[0])
        z_value2 = self.geohash.point_to_z(window[3], window[1])
        # 2. get block1 and block2 by sbrin,
        # and validate the relation in spatial between query window and blocks[block1, block2]
        lm_info_list = self.sbrin.range_query_old(z_value1, z_value2, region)
        result = []
        # 3. for different relation, use different method to handle the points
        for lm_info in lm_info_list:
            # 0 2 1 3的顺序是按照频率降序
            if lm_info[0][0] == 0:  # no relation
                continue
            else:
                if lm_info[2][0] is None:  # block is None
                    continue
                # 3.1 if window contain the block, add all the items into results
                if lm_info[0][0] == 2:  # window contain block
                    result.extend(list(range(lm_info[2][0], lm_info[2][1] + 1)))
                # 3.2 if window intersect or within the block
                else:
                    # 3.2.1 get the min_z/max_z of intersect part
                    lm = self.gm_dict[lm_info[1]]
                    if lm_info[0][0] == 1:  # intersect
                        z_value1 = self.geohash.point_to_z(lm_info[0][1].left, lm_info[0][1].bottom)
                        z_value2 = self.geohash.point_to_z(lm_info[0][1].right, lm_info[0][1].up)
                    # 3.2.2 get the min_index/max_index by nn predict and biased search
                    pre1 = lm.predict(z_value1)
                    pre2 = lm.predict(z_value2)
                    min_err = lm.min_err
                    max_err = lm.max_err
                    left_bound1 = max(round(pre1 - max_err), lm_info[2][0])
                    right_bound1 = min(round(pre1 - min_err), lm_info[2][1])
                    index_left = biased_search(self.index_list, z_value1, int(pre1), left_bound1, right_bound1)
                    if z_value1 == z_value2:
                        if len(index_left) > 0:
                            result.extend(index_left)
                    else:
                        index_left = left_bound1 if len(index_left) == 0 else min(index_left)
                        left_bound2 = max(round(pre2 - max_err), lm_info[2][0])
                        right_bound2 = min(round(pre2 - min_err), lm_info[2][1])
                        index_right = biased_search(self.index_list, z_value2, int(pre2), left_bound2, right_bound2)
                        index_right = right_bound2 if len(index_right) == 0 else max(index_right)
                        # 3.2.3 filter all the point of scope[min_index/max_index] by range.contain(point)
                        result.extend([index for index in range(index_left, index_right + 1)
                                       if region.contain_and_border_by_list(self.point_list[index])])

        return result

    def range_query_old(self, windows):
        return [self.range_query_single_old(window) for window in windows]

    def range_query_single(self, window):
        """
        query index by x1/y1/x2/y2 window
        1. compute z from window_left and window_right
        2. get all relative blocks with index and relationship
        3. get min_z and max_z of every block for different relation
        4. predict min_index/max_index by nn
        5. filter all the point of scope[min_index/max_index] by range.contain(point)
        主要耗时间：sbrin.range_query.ranges_by_int/nn predict/精确过滤: 307mil/145mil/359mil
        """
        if window[0] == window[1] and window[2] == window[3]:
            return self.point_query_single([window[2], window[0]])
        # 1. compute z of window_left and window_right
        z_value1 = self.geohash.point_to_z(window[2], window[0])
        z_value2 = self.geohash.point_to_z(window[3], window[1])
        # 2. get all relative blocks with index and relationship
        blk_index_list = self.sbrin.range_query(z_value1, z_value2)
        result = []
        # 3. get min_z and max_z of every block for different relation
        position_func_list = [lambda index: (None, None, None),
                              lambda index: (  # right
                                  None,
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up),
                                  lambda x: window[3] >= x[0]),
                              lambda index: (  # left
                                  self.geohash.point_to_z(window[2], self.sbrin.blkregs[index].bottom),
                                  None,
                                  lambda x: window[2] <= x[0]),
                              lambda index: (  # left-right
                                  self.geohash.point_to_z(window[2], self.sbrin.blkregs[index].bottom),
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up),
                                  lambda x: window[2] <= x[0] <= window[3]),
                              lambda index: (  # up
                                  None,
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1]),
                                  lambda x: window[1] >= x[1]),
                              lambda index: (  # up-right
                                  None,
                                  z_value2,
                                  lambda x: window[3] >= x[0] and window[1] >= x[1]),
                              lambda index: (  # up-left
                                  self.geohash.point_to_z(window[2], self.sbrin.blkregs[index].bottom),
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1]),
                                  lambda x: window[2] <= x[0] and window[1] >= x[1]),
                              lambda index: (  # up-left-right
                                  self.geohash.point_to_z(window[2], self.sbrin.blkregs[index].bottom),
                                  z_value2,
                                  lambda x: window[2] <= x[0] <= window[3] and window[1] >= x[1]),
                              lambda index: (  # bottom
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  None,
                                  lambda x: window[0] <= x[1]),
                              lambda index: (  # bottom-right
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up),
                                  lambda x: window[3] >= x[0] and window[0] <= x[1]),
                              lambda index: (  # bottom-left
                                  z_value1,
                                  None,
                                  lambda x: window[2] <= x[0] and window[0] <= x[1]),
                              lambda index: (  # bottom-left-right
                                  z_value1,
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up),
                                  lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1]),
                              lambda index: (  # bottom-up
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1]),
                                  lambda x: window[0] <= x[1] <= window[1]),
                              lambda index: (  # bottom-up-right
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  z_value2,
                                  lambda x: window[3] >= x[0] and window[0] <= x[1] <= window[1]),
                              lambda index: (  # bottom-up-left
                                  z_value1,
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1]),
                                  lambda x: window[2] <= x[0] and window[0] <= x[1] <= window[1]),
                              lambda index: (  # bottom-up-left-right
                                  z_value1,
                                  z_value2,
                                  lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1] <= window[1])]
        for blk_index in blk_index_list:
            if self.sbrin.blknums[blk_index] == 0:  # block is None
                continue
            position = blk_index_list[blk_index]
            indexes = self.sbrin.blkindexes[blk_index]
            if position == 0:  # window contain block
                result.extend(list(range(indexes[0], indexes[1] + 1)))
            else:
                # if-elif-else->lambda, 30->4
                z_value_new1, z_value_new2, compare_func = position_func_list[position](blk_index)
                lm = self.gm_dict[blk_index]
                min_err = lm.min_err
                max_err = lm.max_err
                # 4 predict min_index/max_index by nn
                if z_value_new1 is not None:
                    pre1 = lm.predict(z_value_new1)
                    left_bound1 = max(round(pre1 - max_err), indexes[0])
                    right_bound1 = min(round(pre1 - min_err), indexes[1])
                    index_left = min(biased_search_almost(self.index_list, z_value_new1, int(pre1), left_bound1,
                                                          right_bound1))

                else:
                    index_left = indexes[0]
                if z_value_new2 is not None:
                    pre2 = lm.predict(z_value_new2)
                    left_bound2 = max(round(pre2 - max_err), indexes[0])
                    right_bound2 = min(round(pre2 - min_err), indexes[1])
                    index_right = max(biased_search_almost(self.index_list, z_value_new2, int(pre2), left_bound2,
                                                           right_bound2))

                else:
                    index_right = indexes[1]
                # 5 filter all the point of scope[min_index/max_index] by range.contain(point)
                # 优化: region.contain->compare_func不同位置的点做不同的判断: 638->474mil
                result.extend([index for index in range(index_left, index_right + 1)
                               if compare_func(self.point_list[index])])
        return result

    def knn_query_single(self, knn):
        """
        query index by x1/y1/n knn
        1. get the nearest index of query point
        2. get the nn points to create range query window
        3 filter point by distance
        主要耗时间：sbrin.knn_query.ranges_by_int/nn predict/精确过滤: 4.7mil/21mil/14.4mil
        """
        n = knn[2]
        # 1. get the nearest index of query point
        qp_z = self.geohash.point_to_z(knn[0], knn[1])
        qp_blk = self.sbrin.point_query(qp_z)
        qp_model = self.gm_dict[qp_blk]
        qp_model_indexes = self.sbrin.blkindexes[qp_blk]
        # if model is None, qp_index = the max index of the prefix blk
        if qp_model is None:
            while qp_model is None and qp_blk >= 0:
                qp_blk -= 1
            if qp_blk == -1:
                query_point_index = 0
            else:
                query_point_index = self.sbrin.blkindexes[qp_blk][1]
        # if model is not None, qp_index = point_query(z)
        else:
            pre, min_err, max_err = qp_model.predict(qp_z), qp_model.min_err, qp_model.max_err
            left_bound = max(round(pre - max_err), qp_model_indexes[0])
            right_bound = min(round(pre - min_err), qp_model_indexes[1])
            query_point_index = biased_search_almost(self.index_list, qp_z, int(pre), left_bound, right_bound)[0]
        # 2. get the nn points to create range query window
        tp_list = [[Point.distance_pow_point_list(knn, self.point_list[i]), i]
                   for i in range(query_point_index - n, query_point_index + n + 1)]
        tp_list = sorted(tp_list)[:n]
        max_dist = tp_list[-1][0]
        if max_dist == 0:
            return [tp[1] for tp in tp_list]
        max_dist_pow = max_dist ** 0.5
        window = [knn[1] - max_dist_pow, knn[1] + max_dist_pow, knn[0] - max_dist_pow, knn[0] + max_dist_pow]
        z_value1 = self.geohash.point_to_z(window[2], window[0])
        z_value2 = self.geohash.point_to_z(window[3], window[1])
        tp_window_blkes = self.sbrin.knn_query(z_value1, z_value2, knn)
        position_func_list = [lambda index: (None, None),  # window contain block
                              lambda index: (  # right
                                  None,
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up)),
                              lambda index: (  # left
                                  self.geohash.point_to_z(window[2], self.sbrin.blkregs[index].bottom),
                                  None),
                              None,  # left-right
                              lambda index: (  # up
                                  None,
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1])),
                              lambda index: (  # up-right
                                  None,
                                  z_value2),
                              lambda index: (  # up-left
                                  self.geohash.point_to_z(window[2], self.sbrin.blkregs[index].bottom),
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1])),
                              lambda index: (None, None),  # up-left-right
                              lambda index: (  # bottom
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  None),
                              lambda index: (  # bottom-right
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up)),
                              lambda index: (  # bottom-left
                                  z_value1,
                                  None),
                              lambda index: (  # bottom-left-right
                                  z_value1,
                                  self.geohash.point_to_z(window[3], self.sbrin.blkregs[index].up)),
                              None,
                              lambda index: (  # bottom-up-right
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].left, window[0]),
                                  z_value2),
                              lambda index: (  # bottom-up-left
                                  z_value1,
                                  self.geohash.point_to_z(self.sbrin.blkregs[index].right, window[1])),
                              lambda index: (  # bottom-up-left-right
                                  z_value1,
                                  z_value2)]
        tp_list = []
        for tp_window_blk in tp_window_blkes:
            if tp_window_blk[2] > max_dist:
                break
            tp_window_blk_index = tp_window_blk[0]
            if self.sbrin.blknums[tp_window_blk_index] == 0:  # block is None
                continue
            indexes = self.sbrin.blkindexes[tp_window_blk_index]
            z_value_new1, z_value_new2 = position_func_list[tp_window_blk[1]](tp_window_blk_index)
            lm = self.gm_dict[tp_window_blk_index]
            min_err = lm.min_err
            max_err = lm.max_err
            if z_value_new1 is not None:
                pre1 = lm.predict(z_value_new1)
                left_bound1 = max(round(pre1 - max_err), indexes[0])
                right_bound1 = min(round(pre1 - min_err), indexes[1])
                index_left = min(biased_search_almost(self.index_list, z_value_new1, int(pre1), left_bound1,
                                                      right_bound1))
            else:
                index_left = indexes[0]
            if z_value_new2 is not None:
                pre2 = lm.predict(z_value_new2)
                left_bound2 = max(round(pre2 - max_err), indexes[0])
                right_bound2 = min(round(pre2 - min_err), indexes[1])
                index_right = max(biased_search_almost(self.index_list, z_value_new2, int(pre2), left_bound2,
                                                       right_bound2))

            else:
                index_right = indexes[1]
            # 3 filter point by distance
            tp_list.extend([[Point.distance_pow_point_list(knn, self.point_list[i]), i]
                            for i in range(index_left, index_right + 1)])
            tp_list = sorted(tp_list)[:n]
            max_dist = tp_list[-1][0]
        return [tp[1] for tp in tp_list]


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
        elif isinstance(obj, Geohash):
            return obj.save_to_dict()
        elif isinstance(obj, GeoHashModelIndex):
            return obj.save_to_dict()
        elif isinstance(obj, AbstractNN):
            return obj.__dict__
        elif isinstance(obj, SBRIN):
            return obj.save_to_dict()
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
        elif d.__contains__("name") and d["name"] == "Geohash":
            t = Geohash.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "GeoHash Model Index":
            t = GeoHashModelIndex.init_by_dict(d)
        elif len(d.keys()) == 9 and d.__contains__("version"):
            t = SBRIN.init_by_dict(d)
        else:
            t = d
        return t


# @profile(precision=8)
def main():
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
        index.build(data=train_set_xy, threshold_number=1000, data_precision=6, region=Region(40, 42, -75, -73),
                    use_threshold=False,
                    threshold=20,
                    core=[1, 128, 1],
                    train_step=500,
                    batch_size=1024,
                    learning_rate=0.01,
                    retrain_time_limit=20,
                    thread_pool_size=1,
                    load_data=False,
                    save_nn=True)
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
    path = '../../data/trip_data_1_knn_query.csv'
    knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    profile = line_profiler.LineProfiler(index.knn_query_single)
    profile.enable()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    profile.disable()
    profile.print_stats()
    search_time = (end_time - start_time) / len(knn_query_list)
    print("KNN query time ", search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
