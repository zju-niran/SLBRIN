import logging
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region
from src.spatial_index.zm_index import ZMIndex

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/zmindex/stage2_num"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    # data_path = '../../data/index/trip_data_1_filter_10w_sorted.npy'
    data_path = '../../data/index/trip_data_1_filter_sorted.npy'
    stage2_nums = [100, 1000, 10000]
    for stage2_num in stage2_nums:
        model_path = "model/zmindex/stage2_num/%s/" % stage2_num
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)
        index = ZMIndex(model_path=model_path)
        index_name = index.name
        logging.info("*************start %s************" % model_path)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)
        index.build(data_list=data_list,
                    data_precision=6,
                    region=Region(40, 42, -75, -73),
                    use_thresholds=[True, True],
                    thresholds=[0, 0],
                    stages=[1, stage2_num],
                    cores=[[1, 128, 1], [1, 128, 1]],
                    train_steps=[5000, 5000],
                    batch_nums=[64, 64],
                    learning_rates=[0.1, 0.1],
                    retrain_time_limits=[5, 2],
                    thread_pool_size=10,
                    save_nn=True,
                    weight=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        logging.info("Index size: %s" % index.size())
        stage1_model_precision = index.rmi[0][0].max_err - index.rmi[0][0].min_err
        logging.info("Stage1 model precision: %s" % stage1_model_precision)
        stage2_model_precisions = [(model.max_err - model.min_err)
                                   for model in index.rmi[1] if model is not None]
        stage2_model_precisions_avg = sum(stage2_model_precisions) / len(stage2_model_precisions)
        logging.info("Stage2 model precision avg: %s" % stage2_model_precisions_avg)
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
