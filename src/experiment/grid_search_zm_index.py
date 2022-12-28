import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/SBRIN')
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
    stage2_nums = [10000]
    data_distributions = [Distribution.NORMAL_SORTED]
    for data_distribution in data_distributions:
        for stage2_num in stage2_nums:
            model_path = "model/zmindex/%s/stage2_num_%s" % (data_distribution.name, stage2_num)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            index = ZMIndex(model_path=model_path)
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            build_data_list = load_data(data_distribution, 0)
            index.build(data_list=build_data_list,
                        is_sorted=True,
                        data_precision=data_precision[data_distribution],
                        region=data_region[data_distribution],
                        is_new=True,
                        is_simple=False,
                        is_gpu=True,
                        weight=1,
                        stages=[1, stage2_num],
                        cores=[[1, 128], [1, 128]],
                        train_steps=[5000, 5000],
                        batch_nums=[64, 64],
                        learning_rates=[0.1, 0.1],
                        use_thresholds=[False, True],
                        thresholds=[0, 0],
                        retrain_time_limits=[3, 3],
                        thread_pool_size=8)
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            index.model_clear()
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            io_cost = index.io_cost
            stage1_model_precision = index.rmi[0][0].model.max_err - index.rmi[0][0].model.min_err
            logging.info("Stage1 model precision: %s" % stage1_model_precision)
            stage2_model_precisions = [(node.model.max_err - node.model.min_err)
                                       for node in index.rmi[1] if node.model is not None]
            stage2_model_num = len(stage2_model_precisions)
            stage2_model_precisions_avg = sum(stage2_model_precisions) / stage2_model_num
            logging.info("Stage2 model precision avg: %s" % stage2_model_precisions_avg)
            logging.info("Stage2 model number: %s" % stage2_model_num)
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
