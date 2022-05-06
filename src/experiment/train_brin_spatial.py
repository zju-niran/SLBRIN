import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/st-learned-index')
from src.experiment.common_utils import Distribution, load_data, load_query
from src.spatial_index.brin_spatial import BRINSpatial

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/brinspatial"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    # data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    pprs = [64, 128, 256, 512, 1024, 2048]
    for data_distribution in data_distributions:
        logging.info("*************start %s************" % data_distribution)
        for ppr in pprs:
            model_path = "model/brinspatial/%s/ppr_%s" % (data_distribution.name, ppr)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            index = BRINSpatial(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            data_list = load_data(data_distribution)
            index.build(data_list=data_list,
                        pages_per_range=ppr)
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            logging.info("Index size: %s" % index.size())
            path = '../../data/query/point_query.npy'
            point_query_list = load_query(data_distribution, "point").tolist()
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            range_query_list = load_query(data_distribution, "range").tolist()
            for i in range(len(range_query_list) // 1000):
                tmp_range_query_list = range_query_list[i * 1000:(i + 1) * 1000]
                range_ratio = tmp_range_query_list[0][-1]
                start_time = time.time()
                index.test_range_query(tmp_range_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / 1000
                logging.info("Range query ratio: %s" % range_ratio)
                logging.info("Range query time: %s" % search_time)
