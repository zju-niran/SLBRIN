import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/st-learned-index')
from src.experiment.common_utils import Distribution, load_data, data_region, data_precision, load_query
from src.spatial_index.pr_quad_tree import PRQuadTree

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/prquadtree"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    ns = [50, 100, 250, 500]
    for data_distribution in data_distributions:
        for n in ns:
            model_path = "model/prquadtree/%s/n_%s" % (data_distribution.name, n)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            index = PRQuadTree(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            data_list = load_data(data_distribution)
            index.build(data_list=data_list,
                        region=data_region[data_distribution],
                        threshold_number=n,
                        data_precision=data_precision[data_distribution])
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            logging.info("IO cost: %s" % index.io())
            point_query_list = load_query(data_distribution, "point").tolist()
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            range_query_list = load_query(data_distribution, "range").tolist()
            for i in range(len(range_query_list) // 1000):
                tmp_range_query_list = range_query_list[i * 1000:(i + 1) * 1000]
                start_time = time.time()
                index.test_range_query(tmp_range_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / 1000
                logging.info("Range query time: %s" % search_time)
            knn_query_list = load_query(data_distribution, "knn").tolist()
            for i in range(len(knn_query_list) // 1000):
                tmp_knn_query_list = knn_query_list[i * 1000:(i + 1) * 1000]
                start_time = time.time()
                index.test_knn_query(tmp_knn_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / 1000
                logging.info("KNN query time: %s" % search_time)
