import logging
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.experiment.common_utils import Distribution, load_data, data_region, data_precision
from src.spatial_index.zm_index import ZMIndex

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/zmindex"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    # data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    stage2_nums = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
    for data_distribution in data_distributions:
        for stage2_num in stage2_nums:
            model_path = "model/zmindex/%s/stage2_num_%s/" % (data_distribution.name, stage2_num)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            index = ZMIndex(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            data_list = load_data(data_distribution)
            index.build(data_list=data_list,
                        data_precision=data_precision[data_distribution],
                        region=data_region[data_distribution],
                        use_thresholds=[True, True],
                        thresholds=[0, 0],
                        stages=[1, stage2_num],
                        cores=[[1, 128, 1], [1, 128, 1]],
                        train_steps=[5000, 5000],
                        batch_nums=[64, 64],
                        learning_rates=[0.1, 0.1],
                        retrain_time_limits=[5, 5],
                        thread_pool_size=12,
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
            stage2_model_num = len(stage2_model_precisions)
            stage2_model_precisions_avg = sum(stage2_model_precisions) / stage2_model_num
            logging.info("Stage2 model precision avg: %s" % stage2_model_precisions_avg)
            logging.info("Stage2 model number: %s" % stage2_model_num)
            path = '../../data/query/point_query_10w.npy'
            point_query_list = np.load(path, allow_pickle=True).tolist()
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            path = '../../data/query/range_query_10w.npy'
            range_query_list = np.load(path, allow_pickle=True).tolist()
            for i in range(len(range_query_list) // 100):
                tmp_range_query_list = range_query_list[i * 100:(i + 1) * 100]
                range_ratio = tmp_range_query_list[0][4]
                start_time = time.time()
                index.test_range_query(tmp_range_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / 100
                logging.info("Range query ratio:  %s" % range_ratio)
                logging.info("Range query time:  %s" % search_time)
