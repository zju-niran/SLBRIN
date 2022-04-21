import logging
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.r_tree import RTree

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/rtree/fill_factor"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    # data_path = '../../data/table/trip_data_1_filter_10w.npy'
    data_path = '../../data/table/trip_data_1_filter.npy'
    fill_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
    for fill_factor in fill_factors:
        model_path = "model/rtree/fill_factor/%s" % fill_factor
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        index = RTree(model_path=model_path)
        index_name = index.name
        logging.info("*************start %s************" % model_path)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)[:, [10, 11, -1]]
        index.build(data_list=data_list,
                    fill_factor=fill_factor,
                    leaf_node_capacity=113,
                    non_leaf_node_capacity=113,
                    buffering_capacity=None)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        logging.info("Index size: %s" % index.size())
        path = '../../data/query/point_query.npy'
        point_query_list = np.load(path, allow_pickle=True).tolist()
        start_time = time.time()
        index.test_point_query(point_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(point_query_list)
        logging.info("Point query time: %s" % search_time)
        path = '../../data/query/range_query.npy'
        range_query_list = np.load(path, allow_pickle=True).tolist()
        for i in range(len(range_query_list) // 1000):
            tmp_range_query_list = range_query_list[i * 1000:(i + 1) * 1000]
            range_ratio = tmp_range_query_list[0][4]
            start_time = time.time()
            index.test_range_query(tmp_range_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / 1000
            logging.info("Range query ratio:  %s" % range_ratio)
            logging.info("Range query time:  %s" % search_time)
        path = '../../data/query/knn_query.npy'
        knn_query_list = np.load(path, allow_pickle=True).tolist()
        for i in range(len(knn_query_list) // 1000):
            tmp_knn_query_list = knn_query_list[i * 1000:(i + 1) * 1000]
            n = tmp_knn_query_list[0][2]
            start_time = time.time()
            index.test_knn_query(tmp_knn_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / 1000
            logging.info("KNN query n:  %s" % n)
            logging.info("KNN query time:  %s" % search_time)
