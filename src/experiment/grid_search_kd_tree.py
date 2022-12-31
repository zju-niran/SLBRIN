import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/SBRIN')
from src.experiment.common_utils import Distribution, load_data, load_query
from src.spatial_index.kd_tree import KDTree

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/kdtree"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    for data_distribution in data_distributions:
        model_path = "model/kdtree/%s" % data_distribution.name
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        index = KDTree(model_path=model_path)
        logging.info("*************start %s************" % model_path)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        structure_size, ie_size = index.size()
        logging.info("Structure size: %s" % structure_size)
        logging.info("Index entry size: %s" % ie_size)
        io_cost = index.io_cost
        logging.info("IO cost: %s" % io_cost)
        point_query_list = load_query(data_distribution, 0).tolist()
        start_time = time.time()
        index.test_point_query(point_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(point_query_list)
        logging.info("Point query time: %s" % search_time)
        logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
        io_cost = index.io_cost
        range_query_list = load_query(data_distribution, 1).tolist()
        for i in range(len(range_query_list) // 1000):
            tmp_range_query_list = range_query_list[i * 1000:(i + 1) * 1000]
            start_time = time.time()
            index.test_range_query(tmp_range_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(tmp_range_query_list)
            logging.info("Range query time: %s" % search_time)
            logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(tmp_range_query_list)))
            io_cost = index.io_cost
        knn_query_list = load_query(data_distribution, 2).tolist()
        for i in range(len(knn_query_list) // 1000):
            tmp_knn_query_list = knn_query_list[i * 1000:(i + 1) * 1000]
            start_time = time.time()
            index.test_knn_query(tmp_knn_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(tmp_knn_query_list)
            logging.info("KNN query time: %s" % search_time)
            logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(tmp_knn_query_list)))
            io_cost = index.io_cost

