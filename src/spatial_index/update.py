import logging
import os
import sys
import time

from src.spatial_index.zm_index import ZMIndex

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region
from src.experiment.common_utils import load_data, Distribution, load_query, data_precision, data_region


class dataset:
    def __init__(self):
        return


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "update/normal/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    data_distribution = Distribution.UNIFORM_10W
    # 使用构建数据构建ZM
    index = ZMIndex(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        # 加载构建数据
        build_data_list = load_data(data_distribution, 0)
        # 使用构建数据构建ZM
        index.build(data_list=build_data_list,
                    is_sorted=False,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    is_new=True,
                    is_simple=False,
                    is_gpu=True,
                    weight=1,
                    stages=[1],
                    cores=[[1, 128]],
                    train_steps=[5000],
                    batch_nums=[64],
                    learning_rates=[0.1],
                    use_thresholds=[False],
                    thresholds=[5],
                    retrain_time_limits=[2],
                    thread_pool_size=6)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    logging.info("IO cost: %s" % index.io())
    # 加载更新数据
    update_data_list = load_data(data_distribution, 1)
    update_data_list_len = len(update_data_list)
    # 划分更新数据
    batch_num = 5
    batch_size = len(update_data_list) // batch_num + 1
    cur_index = 0
    update_data_lists = []
    while cur_index + batch_size <= update_data_list_len:
        update_data_lists.append(update_data_list[cur_index:cur_index + batch_size])
        cur_index += batch_size
    if cur_index < update_data_list_len:
        update_data_lists.append(update_data_list[cur_index:])
    # 记录更新时间、索引体积、磁盘IO和查询时间
    for j in range(5):
        logging.info("Update idx: %s" % j)
        start_time = time.time()
        index.insert(update_data_lists[j])
        index.save()
        end_time = time.time()
        logging.info("Update time: %s" % (end_time - start_time))
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
        start_time = time.time()
        index.test_range_query(range_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(range_query_list)
        logging.info("Range query time: %s" % search_time)
        knn_query_list = load_query(data_distribution, "knn").tolist()
        start_time = time.time()
        index.test_knn_query(knn_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(knn_query_list)
        logging.info("KNN query time: %s" % search_time)
