import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/st-learned-index')
from src.experiment.common_utils import Distribution, load_data, data_region, data_precision, load_query
from src.spatial_index.zm_index import ZMIndex

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/zmindex"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.NYCT_SORTED, Distribution.NORMAL_SORTED, Distribution.UNIFORM_SORTED]
    stage2_nums = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
    for data_distribution in data_distributions:
        for stage2_num in stage2_nums:
            model_path = "model/zmindex/%s/stage2_num_%s" % (data_distribution.name, stage2_num)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            index = ZMIndex(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            data_list = load_data(data_distribution)
            index.build(data_list=data_list,
                        is_sorted=True,
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
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            logging.info("IO cost: %s" % index.io())
            stage1_model_precision = index.rmi[0][0].max_err - index.rmi[0][0].min_err
            logging.info("Stage1 model precision: %s" % stage1_model_precision)
            stage2_model_precisions = [(model.max_err - model.min_err)
                                       for model in index.rmi[1] if model is not None]
            stage2_model_num = len(stage2_model_precisions)
            stage2_model_precisions_avg = sum(stage2_model_precisions) / stage2_model_num
            logging.info("Stage2 model precision avg: %s" % stage2_model_precisions_avg)
            logging.info("Stage2 model number: %s" % stage2_model_num)
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
