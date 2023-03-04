import logging
import os
import shutil
import sys
import time

from src.spatial_index.zm_index_ipi import ZMIndexInPlaceInsert

sys.path.append('/home/zju/wlj/SLBRIN')
from src.spatial_index.tsusli import TSUSLI
from src.spatial_index.zm_index_di import ZMIndexDeltaInsert
from src.experiment.common_utils import load_data, Distribution, copy_dirs, load_query

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    origin_path = "model/origin/"
    target_path = "model/compare_tsusli/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.basicConfig(filename=os.path.join(target_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    index_infos = [
        ("dusli", True, True),
        ("ipusli", False, False, 0.2),
        ("ipusli2", False, False, 0.8),
        ("tsusli", False, False, 24, 3, 100),
    ]
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    # data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    for index_info in index_infos:
        for data_distribution in data_distributions:
            # copy the zm_index as the basic
            origin_model_path = origin_path + data_distribution.name
            target_model_path = target_path + data_distribution.name + "/" + index_info[0]
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            copy_dirs(origin_model_path, target_model_path)
            # initial the compared model from the zm_index
            logging.info("*************start %s %s************" % (index_info[0], data_distribution.name))
            start_time = time.time()
            if index_info[0] == "dusli":
                index = ZMIndexDeltaInsert(model_path=target_model_path)
                super(ZMIndexDeltaInsert, index).load()
                index.build_append(time_interval=60 * 60,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   is_retrain=index_info[1],
                                   is_save=index_info[2])
            elif index_info[0] == "tsusli":
                index = TSUSLI(model_path=target_model_path)
                super(TSUSLI, index).load()
                index.build_append(time_interval=60 * 60,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   is_retrain=index_info[1],
                                   is_save=index_info[2],
                                   lag=index_info[3],
                                   predict_step=index_info[4],
                                   cdf_width=index_info[5])
            else:
                index = ZMIndexInPlaceInsert(model_path=target_model_path)
                super(ZMIndexInPlaceInsert, index).load()
                index.build_append(time_interval=60 * 60,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   is_retrain=index_info[1],
                                   is_save=index_info[2],
                                   empty_ratio=index_info[3])
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            io_cost = 0
            logging.info("Model precision avg: %s" % index.model_err())
            point_query_list = load_query(data_distribution, 0).tolist()
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
            io_cost = index.io_cost
            update_data_list = load_data(data_distribution, 1)
            start_time = time.time()
            index.insert(update_data_list)
            end_time = time.time()
            logging.info("Update time: %s" % (end_time - start_time))
            logging.info("Statis list: %s" % index.statistic_list)
            point_query_list = load_query(data_distribution, 0).tolist()
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
            io_cost = index.io_cost
