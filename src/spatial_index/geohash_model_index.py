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
from src.sbrin import SBRIN, RegularPage, MetaPage
from src.spatial_index.common_utils import Region, biased_search, Point, biased_search_almost
from src.spatial_index.geohash_utils import Geohash
from src.spatial_index.spatial_index import SpatialIndex
from src.rmi_keras import TrainedNN, AbstractNN
from src.rmi_keras_simple import TrainedNN as TrainedNN_Simple

"""
代码上和论文的diff:
1. 数据和geohash索引没有分开，而是放在data_list里：理论上，索引中找到key后能找到数据磁盘位置，然后直接读取point，但代码实现不了
为了模拟和贴近寻道的效率，索引上直接放了数据；
索引size = 数据+索引的size - 数据的size + sbrin.json的size；
索引构建时间 = 数据geohash编码时间+sbrin构建时间
"""


# 磁盘io率用扫描过的data_list数据量来表示
# SBRIN是一级索引，geohash只是数据排序和blk range mbr计算的工具
# 对于数据更新
# 策略1：不要搞缓存和留空了，换一种策略，就是定期生成新的索引，数据就按时间插入，生成新的blk range geohash mbr，一定时间后更新整体的数据顺序形成新的SBRIN，再一定时间后合并到老的SBRIN
# 策略2：留空，如果blk range满了，就生成新的blk range，配上原来blk range一样的属性；如果没满，就先放到队尾，并且检查误差（误差不一定变大的）。
class GeoHashModelIndex(SpatialIndex):
    def __init__(self, model_path=None, sbrin=None):
        super(GeoHashModelIndex, self).__init__("GeoHash Model Index")
        self.sbrin = sbrin
        self.data_list = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def build(self, data_list, threshold_number, data_precision, region, use_threshold, threshold, core,
              train_step, batch_size, learning_rate, retrain_time_limit, thread_pool_size, save_nn,
              weight):
        """
        build index
        1. init train geohash->key data from x/y data
        2. create brin index
        3. create learned model(stage=1) for every leaf node
        """
        # 1. init train geohash->key data from x/y data
        # 2. create brin index
        start_time = time.time()
        self.sbrin = SBRIN()
        block_size = 100
        threshold_length = region.get_max_depth_by_region_and_precision(precision=data_precision) * 2
        self.sbrin.build(data_list, threshold_number, threshold_length, region, data_precision, block_size)
        end_time = time.time()
        self.logging.info("Create SBRIN: %s" % (end_time - start_time))
        # reconstruct data
        start_time = time.time()
        result_data_list = []
        for regular_page in self.sbrin.regular_pages:
            result_data_list.extend(data_list[regular_page.key[0]: regular_page.key[1] + 1])
            diff_length = threshold_number - regular_page.number
            result_data_list.extend([None] * diff_length)
            regular_page.key = [threshold_number * (regular_page.itemoffset - 1),
                                threshold_number * (regular_page.itemoffset - 1) + regular_page.number - 1]
        self.data_list = result_data_list
        end_time = time.time()
        self.logging.info("Reconstruct data: %s" % (end_time - start_time))
        # 3. in every part data, create learned model
        start_time = time.time()
        self.build_nn_multiprocess(use_threshold, threshold, core, train_step, batch_size, learning_rate,
                                   retrain_time_limit, thread_pool_size, save_nn, weight)
        end_time = time.time()
        self.logging.info("Create learned model: %s" % (end_time - start_time))

    def build_nn_multiprocess(self, use_threshold, threshold, core, train_step, batch_size, learning_rate,
                              retrain_time_limit, thread_pool_size, save_nn, weight):
        multiprocessing.set_start_method('spawn', force=True)  # 解决CUDA_ERROR_NOT_INITIALIZED报错
        pool = multiprocessing.Pool(processes=thread_pool_size)
        mp_dict = multiprocessing.Manager().dict()
        for regular_page in self.sbrin.regular_pages:
            key_bound = regular_page.key
            if regular_page.number == 0:  # blk range is None
                continue
            inputs = [i[2] for i in self.data_list[key_bound[0]:key_bound[1] + 1]]
            labels = list(range(key_bound[0], key_bound[1] + 1))
            pool.apply_async(self.build_nn, (regular_page.itemoffset - 1, inputs, labels, use_threshold,
                                             threshold, core, train_step, batch_size, learning_rate,
                                             retrain_time_limit, save_nn, weight, mp_dict))
        pool.close()
        pool.join()
        for (key, value) in mp_dict.items():
            self.sbrin.regular_pages[key].model = value

    def build_nn(self, model_key, inputs, labels, use_threshold, threshold, core, train_step, batch_size,
                 learning_rate, retrain_time_limit, save_nn, weight, tmp_dict=None):
        # train model
        if save_nn is False:
            tmp_index = TrainedNN_Simple(self.model_path, model_key, inputs, labels, core, train_step, batch_size,
                                         learning_rate, weight)
            tmp_index.train()
        else:
            tmp_index = TrainedNN(self.model_path, str(model_key), inputs, labels,
                                  use_threshold,
                                  threshold,
                                  core,
                                  train_step,
                                  batch_size,
                                  learning_rate,
                                  retrain_time_limit,
                                  weight)
            tmp_index.train()
        # get parameters in model (weight matrix and bias matrix)
        abstract_index = AbstractNN(tmp_index.weights,
                                    core,
                                    tmp_index.train_x_min,
                                    tmp_index.train_x_max,
                                    tmp_index.train_y_min,
                                    tmp_index.train_y_max,
                                    tmp_index.min_err,
                                    tmp_index.max_err)
        del tmp_index
        gc.collect()
        tmp_dict[model_key] = abstract_index

    def save(self):
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        with open(self.model_path + 'sbrin.json', "w") as f:
            json.dump(self, f, cls=MyEncoder, ensure_ascii=False)
        np.save(self.model_path + 'data_list.npy', np.array(self.data_list))

    def load(self):
        with open(self.model_path + 'sbrin.json', "r") as f:
            sbrin = json.load(f, cls=MyDecoder)
            self.sbrin = sbrin.sbrin
            self.data_list = np.load(self.model_path + 'data_list.npy', allow_pickle=True).tolist()
            del sbrin

    def size(self):
        return os.path.getsize(os.path.join(self.model_path, "sbrin.json"))

    @staticmethod
    def init_by_dict(d: dict):
        return GeoHashModelIndex(model_path=d['model_path'], sbrin=d['sbrin'])

    def save_to_dict(self):
        return {
            'name': self.name,
            'sbrin': self.sbrin,
            'model_path': self.model_path
        }

    def point_query_single(self, point):
        """
        query key by x/y point
        1. compute geohash from x/y of points
        2. find blk range within geohash by sbrin.point_query
        3. predict by leaf model and create key scope [pre - min_err, pre + max_err]
        4. binary search in scope
        """
        # 1. compute geohash from x/y of point
        gh = self.sbrin.meta_page.geohash.encode(point[0], point[1])
        # 2. find blk range within geohash by sbrin.point_query
        blk_range = self.sbrin.regular_pages[self.sbrin.point_query(gh)]
        if blk_range.model is None:
            return None
        else:
            # 3. predict by leaf model and create key scope [pre - min_err, pre + max_err]
            pre, min_err, max_err = blk_range.model.predict(gh), blk_range.model.min_err, blk_range.model.max_err
            # 4. binary search in scope
            # 优化: round->int:2->1
            return biased_search(self.data_list, 2, gh, int(pre),
                                 max(round(pre - max_err), blk_range.key[0]),
                                 min(round(pre - min_err), blk_range.key[1]))

    def range_query_single_old(self, window):
        """
        query key by x1/y1/x2/y2 window
        1. compute geohash from window_left and window_right
        2. get all the blk range and its relationship with window between geohash1/geohash2 by sbrin.range_query
        3. for different relation, use different method to handle the points
        3.1 if window contain the blk range, add all the items into results
        3.2 if window intersect or within the blk range
        3.2.1 get the min_geohash/max_geohash of intersect part
        3.2.2 get the min_key/max_key by nn predict and biased search
        3.2.3 filter all the point of scope[min_key/max_key] by range.contain(point)
        主要耗时间：两次geohash, predict和最后的精确过滤，0.1, 0.1 , 0.6
        # TODO: 由于build sbrin的时候region移动了，导致这里的查询不准确了
        """
        region = Region(window[0], window[1], window[2], window[3])
        # 1. compute geohash of window_left and window_right
        gh1 = self.sbrin.meta_page.geohash.encode(window[2], window[0])
        gh2 = self.sbrin.meta_page.geohash.encode(window[3], window[1])
        # 2. get all the blk range and its relationship with window between geohash1/geohash2 by sbrin.range_query
        blk_range_list = self.sbrin.range_query_old(gh1, gh2, region)
        result = []
        # 3. for different relation, use different method to handle the points
        for blk_range in blk_range_list:
            # 0 2 1 3的顺序是按照频率降序
            if blk_range[0][0] == 0:  # no relation
                continue
            else:
                if blk_range[1].model is None:  # blk range is None
                    continue
                # 3.1 if window contain the blk range, add all the items into results
                if blk_range[0][0] == 2:  # window contain blk range
                    result.extend(list(range(blk_range[1].key[0], blk_range[1].key[1] + 1)))
                # 3.2 if window intersect or within the blk range
                else:
                    # 3.2.1 get the min_geohash/max_geohash of intersect part
                    model = blk_range[1].model
                    if blk_range[0][0] == 1:  # intersect
                        gh1 = self.sbrin.meta_page.geohash.encode(blk_range[0][1].left, blk_range[0][1].bottom)
                        gh2 = self.sbrin.meta_page.geohash.encode(blk_range[0][1].right, blk_range[0][1].up)
                    # 3.2.2 get the min_key/max_key by nn predict and biased search
                    pre1 = model.predict(gh1)
                    pre2 = model.predict(gh2)
                    min_err = model.min_err
                    max_err = model.max_err
                    l_bound1 = max(round(pre1 - max_err), blk_range[1].key[0])
                    r_bound1 = min(round(pre1 - min_err), blk_range[1].key[1])
                    key_left = biased_search(self.data_list, 2, gh1, int(pre1), l_bound1, r_bound1)
                    if gh1 == gh2:
                        if len(key_left) > 0:
                            result.extend(key_left)
                    else:
                        key_left = l_bound1 if len(key_left) == 0 else min(key_left)
                        l_bound2 = max(round(pre2 - max_err), blk_range[1].key[0])
                        r_bound2 = min(round(pre2 - min_err), blk_range[1].key[1])
                        key_right = biased_search(self.data_list, 2, gh2, int(pre2), l_bound2, r_bound2)
                        key_right = r_bound2 if len(key_right) == 0 else max(key_right)
                        # 3.2.3 filter all the point of scope[min_key/max_key] by range.contain(point)
                        result.extend([key for key in range(key_left, key_right + 1)
                                       if region.contain_and_border_by_list(self.data_list[key])])
        return result

    def range_query_old(self, windows):
        return [self.range_query_single_old(window) for window in windows]

    def range_query_single(self, window):
        """
        query key by x1/y1/x2/y2 window
        1. compute geohash from window_left and window_right
        2. get all relative blk ranges with key and relationship
        3. get min_geohash and max_geohash of every blk range for different relation
        4. predict min_key/max_key by nn
        5. filter all the point of scope[min_key/max_key] by range.contain(point)
        主要耗时间：sbrin.range_query.ranges_by_int/nn predict/精确过滤: 307mil/145mil/359mil
        """
        if window[0] == window[1] and window[2] == window[3]:
            return self.point_query_single([window[2], window[0]])
        # 1. compute geohash of window_left and window_right
        gh1 = self.sbrin.meta_page.geohash.encode(window[2], window[0])
        gh2 = self.sbrin.meta_page.geohash.encode(window[3], window[1])
        # 2. get all relative blk ranges with key and relationship
        blk_list = self.sbrin.range_query(gh1, gh2)
        result = []
        # 3. get min_geohash and max_geohash of every blk range for different relation
        position_func_list = [lambda reg: (None, None, None),
                              lambda reg: (  # right
                                  None,
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up),
                                  lambda x: window[3] >= x[0]),
                              lambda reg: (  # left
                                  self.sbrin.meta_page.geohash.encode(window[2], reg.bottom),
                                  None,
                                  lambda x: window[2] <= x[0]),
                              lambda reg: (  # left-right
                                  self.sbrin.meta_page.geohash.encode(window[2], reg.bottom),
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up),
                                  lambda x: window[2] <= x[0] <= window[3]),
                              lambda reg: (  # up
                                  None,
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1]),
                                  lambda x: window[1] >= x[1]),
                              lambda reg: (  # up-right
                                  None,
                                  gh2,
                                  lambda x: window[3] >= x[0] and window[1] >= x[1]),
                              lambda reg: (  # up-left
                                  self.sbrin.meta_page.geohash.encode(window[2], reg.bottom),
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1]),
                                  lambda x: window[2] <= x[0] and window[1] >= x[1]),
                              lambda reg: (  # up-left-right
                                  self.sbrin.meta_page.geohash.encode(window[2], reg.bottom),
                                  gh2,
                                  lambda x: window[2] <= x[0] <= window[3] and window[1] >= x[1]),
                              lambda reg: (  # bottom
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  None,
                                  lambda x: window[0] <= x[1]),
                              lambda reg: (  # bottom-right
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up),
                                  lambda x: window[3] >= x[0] and window[0] <= x[1]),
                              lambda reg: (  # bottom-left
                                  gh1,
                                  None,
                                  lambda x: window[2] <= x[0] and window[0] <= x[1]),
                              lambda reg: (  # bottom-left-right
                                  gh1,
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up),
                                  lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1]),
                              lambda reg: (  # bottom-up
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1]),
                                  lambda x: window[0] <= x[1] <= window[1]),
                              lambda reg: (  # bottom-up-right
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  gh2,
                                  lambda x: window[3] >= x[0] and window[0] <= x[1] <= window[1]),
                              lambda reg: (  # bottom-up-left
                                  gh1,
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1]),
                                  lambda x: window[2] <= x[0] and window[0] <= x[1] <= window[1]),
                              lambda reg: (  # bottom-up-left-right
                                  gh1,
                                  gh2,
                                  lambda x: window[2] <= x[0] <= window[3] and window[0] <= x[1] <= window[1])]
        for blk_key in blk_list:
            blk = self.sbrin.regular_pages[blk_key]
            if blk.model is None:  # blk range is None
                continue
            position = blk_list[blk_key]
            if position == 0:  # window contain blk range
                result.extend(list(range(blk.key[0], blk.key[1] + 1)))
            else:
                # if-elif-else->lambda, 30->4
                gh_new1, gh_new2, compare_func = position_func_list[position](blk.scope)
                model = blk.model
                min_err = model.min_err
                max_err = model.max_err
                # 4 predict min_key/max_key by nn
                if gh_new1 is not None:
                    pre1 = model.predict(gh_new1)
                    l_bound1 = max(round(pre1 - max_err), blk.key[0])
                    r_bound1 = min(round(pre1 - min_err), blk.key[1])
                    key_left = min(biased_search_almost(self.data_list, 2, gh_new1, int(pre1), l_bound1, r_bound1))
                else:
                    key_left = blk.key[0]
                if gh_new2 is not None:
                    pre2 = model.predict(gh_new2)
                    l_bound2 = max(round(pre2 - max_err), blk.key[0])
                    r_bound2 = min(round(pre2 - min_err), blk.key[1])
                    key_right = max(biased_search_almost(self.data_list, 2, gh_new2, int(pre2), l_bound2, r_bound2))
                else:
                    key_right = blk.key[1]
                # 5 filter all the point of scope[min_key/max_key] by range.contain(point)
                # 优化: region.contain->compare_func不同位置的点做不同的判断: 638->474mil
                result.extend([key for key in range(key_left, key_right + 1)
                               if compare_func(self.data_list[key])])
        return result

    def knn_query_single(self, knn):
        """
        query key by x1/y1/n knn
        1. get the nearest key of query point
        2. get the nn points to create range query window
        3. filter point by distance
        主要耗时间：sbrin.knn_query.ranges_by_int/nn predict/精确过滤: 4.7mil/21mil/14.4mil
        """
        k = knn[2]
        # 1. get the nearest key of query point
        qp_g = self.sbrin.meta_page.geohash.encode(knn[0], knn[1])
        qp_blk_key = self.sbrin.point_query(qp_g)
        qp_blk = self.sbrin.regular_pages[qp_blk_key]
        # if blk range is None, qp_key = the max key of the last blk range
        if qp_blk.model is None:
            query_point_key = qp_blk.key[1]
        # if model is not None, qp_key = point_query(geohash)
        else:
            pre, min_err, max_err = qp_blk.model.predict(qp_g), qp_blk.model.min_err, qp_blk.model.max_err
            l_bound = max(round(pre - max_err), qp_blk.key[0])
            r_bound = min(round(pre - min_err), qp_blk.key[1])
            query_point_key = biased_search_almost(self.data_list, 2, qp_g, int(pre), l_bound, r_bound)[0]
        # 2. get the n points to create range query window
        # TODO: 两种策略，一种是左右找一半，但是如果跳跃了，window很大；还有一种是两边找n，减少跳跃，使window变小
        tp_list = [[Point.distance_pow_point_list(knn, self.data_list[query_point_key]), query_point_key]]
        cur_key = query_point_key + 1
        cur_block_key = qp_blk_key
        i = 0
        while i < k - 1:
            if self.data_list[cur_key] is None:
                cur_block_key += 1
                if cur_block_key > self.sbrin.meta_page.size:
                    break
                cur_key = self.sbrin.regular_pages[cur_block_key].key[0]
            else:
                tp_list.append([Point.distance_pow_point_list(knn, self.data_list[cur_key]), cur_key])
                cur_key += 1
                i += 1
        cur_key = query_point_key - 1
        cur_block_key = qp_blk_key
        i = 0
        while i < k - 1:
            if self.data_list[cur_key] is None:
                cur_block_key -= 1
                if cur_block_key < 0:
                    break
                cur_key = self.sbrin.regular_pages[qp_blk_key].key[1]
            else:
                tp_list.append([Point.distance_pow_point_list(knn, self.data_list[cur_key]), cur_key])
                cur_key -= 1
                i += 1
        tp_list = sorted(tp_list)[:k]
        max_dist = tp_list[-1][0]
        if max_dist == 0:
            return [tp[1] for tp in tp_list]
        max_dist_pow = max_dist ** 0.5
        window = [knn[1] - max_dist_pow, knn[1] + max_dist_pow, knn[0] - max_dist_pow, knn[0] + max_dist_pow]
        gh1 = self.sbrin.meta_page.geohash.encode(window[2], window[0])
        gh2 = self.sbrin.meta_page.geohash.encode(window[3], window[1])
        tp_window_blkes = self.sbrin.knn_query(gh1, gh2, knn)
        position_func_list = [lambda reg: (None, None),  # window contain blk range
                              lambda reg: (  # right
                                  None,
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up)),
                              lambda reg: (  # left
                                  self.sbrin.meta_page.geohash.encode(window[2], reg.bottom),
                                  None),
                              None,  # left-right
                              lambda reg: (  # up
                                  None,
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1])),
                              lambda reg: (  # up-right
                                  None,
                                  gh2),
                              lambda reg: (  # up-left
                                  self.sbrin.meta_page.geohash.encode(window[2], reg.bottom),
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1])),
                              lambda reg: (None, None),  # up-left-right
                              lambda reg: (  # bottom
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  None),
                              lambda reg: (  # bottom-right
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up)),
                              lambda reg: (  # bottom-left
                                  gh1,
                                  None),
                              lambda reg: (  # bottom-left-right
                                  gh1,
                                  self.sbrin.meta_page.geohash.encode(window[3], reg.up)),
                              None,
                              lambda reg: (  # bottom-up-right
                                  self.sbrin.meta_page.geohash.encode(reg.left, window[0]),
                                  gh2),
                              lambda reg: (  # bottom-up-left
                                  gh1,
                                  self.sbrin.meta_page.geohash.encode(reg.right, window[1])),
                              lambda reg: (  # bottom-up-left-right
                                  gh1,
                                  gh2)]
        tp_list = []
        for tp_window_blk in tp_window_blkes:
            if tp_window_blk[2] > max_dist:
                break
            blk = self.sbrin.regular_pages[tp_window_blk[0]]
            model = blk.model
            if model is None:  # blk range is None
                continue
            blk_key = blk.key
            gh_new1, gh_new2 = position_func_list[tp_window_blk[1]](blk.scope)
            min_err = model.min_err
            max_err = model.max_err
            if gh_new1 is not None:
                pre1 = model.predict(gh_new1)
                l_bound1 = max(round(pre1 - max_err), blk_key[0])
                r_bound1 = min(round(pre1 - min_err), blk_key[1])
                key_left = min(biased_search_almost(self.data_list, 2, gh_new1, int(pre1), l_bound1, r_bound1))
            else:
                key_left = blk_key[0]
            if gh_new2 is not None:
                pre2 = model.predict(gh_new2)
                l_bound2 = max(round(pre2 - max_err), blk_key[0])
                r_bound2 = min(round(pre2 - min_err), blk_key[1])
                key_right = max(biased_search_almost(self.data_list, 2, gh_new2, int(pre2), l_bound2, r_bound2))

            else:
                key_right = blk_key[1]
            # 3. filter point by distance
            tp_list.extend([[Point.distance_pow_point_list(knn, self.data_list[i]), i]
                            for i in range(key_left, key_right + 1)])
            tp_list = sorted(tp_list)[:k]
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
            return obj.__dict__
        elif isinstance(obj, RegularPage):
            return obj.__dict__
        elif isinstance(obj, MetaPage):
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
        elif d.__contains__("name") and d["name"] == "Geohash":
            t = Geohash.init_by_dict(d)
        elif d.__contains__("name") and d["name"] == "GeoHash Model Index":
            t = GeoHashModelIndex.init_by_dict(d)
        elif d.__contains__("regular_pages"):
            t = SBRIN.init_by_dict(d)
        elif d.__contains__("itemoffset"):
            t = RegularPage.init_by_dict(d)
        elif d.__contains__("pages_per_range"):
            t = MetaPage.init_by_dict(d)
        else:
            t = d
        return t


# @profile(precision=8)
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/trip_data_1_10w_sorted.npy'
    model_path = "model/gm_index_10w_1/"
    index = GeoHashModelIndex(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        data_list = np.load(data_path).tolist()
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(data_list=data_list, threshold_number=1000, data_precision=6, region=Region(40, 42, -75, -73),
                    use_threshold=False,
                    threshold=20,
                    core=[1, 128, 1],
                    train_step=500,
                    batch_size=1024,
                    learning_rate=0.01,
                    retrain_time_limit=20,
                    thread_pool_size=1,
                    save_nn=True,
                    weight=0.01)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        index.save()
    logging.info("Index size: %s" % index.size())
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
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    print("KNN query time ", search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
