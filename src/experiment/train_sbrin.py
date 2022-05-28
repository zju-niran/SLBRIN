import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/st-learned-index')
from src.experiment.common_utils import Distribution, load_query, load_data, data_precision, data_region
from src.spatial_index.sbrin import SBRIN

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/sbrin"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    tn_list = [160000, 80000, 40000, 20000, 10000, 5000]
    for data_distribution in data_distributions:
        for tn in tn_list:
            model_path = "model/sbrin/%s/tn_%s" % (data_distribution.name, tn)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            index = SBRIN(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            data_list = load_data(data_distribution)
            index.build(data_list=data_list,
                        is_sorted=True,
                        threshold_number=tn,
                        data_precision=data_precision[data_distribution],
                        region=data_region[data_distribution],
                        threshold_err=200,
                        threshold_summary=1000,
                        threshold_merge=5,
                        is_new=True,
                        is_simple=False,
                        is_gpu=True,
                        weight=1,
                        core=[1, 128],
                        train_step=5000,
                        batch_num=64,
                        learning_rate=0.1,
                        use_threshold=False,
                        threshold=0,
                        retrain_time_limit=0,
                        thread_pool_size=12)
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            logging.info("IO cost: %s" % index.io())
            model_num = index.meta.last_hr + 1
            logging.info("Model num: %s" % model_num)
            model_precisions = [(hr.model.max_err - hr.model.min_err)
                                for hr in index.history_ranges]
            model_precisions_avg = sum(model_precisions) / model_num
            logging.info("Model precision avg: %s" % model_precisions_avg)
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
