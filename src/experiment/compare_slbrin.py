import logging
import os
import shutil
import time

from src.experiment.common_utils import Distribution, load_data, load_query, copy_dirs
from src.si.brin_spatial import BRINSpatial
from src.si.kd_tree import KDTree
from src.si.pr_quad_tree import PRQuadTree
from src.si.r_tree import RTree
from src.proposed_sli.slbrin import SLBRIN
from src.proposed_sli.slibs import SLIBS
from src.sli.zm_index import ZMIndex

"""
实验探究：对比RT/PRQT/KDT/BRINS/ZM/SLIBS/SLBRIN的更新性能
1. 划分更新数据集为5等份
2. 每份写入后测试索引体积、更新性能和检索性能
"""
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    origin_path = "model/"
    target_path = "model/update/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.basicConfig(filename=os.path.join(target_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    index_infos = [
        (RTree, "rtree/%s_0.8", "rtree/%s"),
        (PRQuadTree, "prquadtree/%s_500", "prquadtree/%s"),
        (KDTree, "kdtree/%s", "kdtree/%s"),
        (BRINSpatial, "brinspatial/%s_SORTED_sorted_64", "brinspatial/%s"),
        (ZMIndex, "zmindex/%s_SORTED_1000", "zmindex/%s"),
        (SLIBS, "slibs/%s_SORTED_1000", "slibs/%s"),
        (SLBRIN, "slbrin/%s_SORTED_10000", "slbrin/%s")]
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    for data_distribution in data_distributions:
        build_data_list = load_data(data_distribution, 0)
        point_query_list = load_query(data_distribution, 0).tolist()
        range_query_list = load_query(data_distribution, 1).tolist()
        knn_query_list = load_query(data_distribution, 2).tolist()
        for index_info in index_infos:
            # 拷贝目标索引磁盘文件
            origin_model_path = origin_path + index_info[1] % data_distribution.name
            target_model_path = target_path + index_info[2] % data_distribution.name
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            copy_dirs(origin_model_path, target_model_path)
            # 加载索引结构到内存
            index = index_info[0](model_path=target_model_path)
            index.load()
            if index_info[0] is SLBRIN:
                index.meta.threshold_err = 1.0
            # 更新数据集一分为五
            update_data_list = load_data(data_distribution, 1)
            update_data_len = len(update_data_list)
            tmp_update_data_list_len = update_data_len // 5 + 1
            tmp_update_data_lists = [update_data_list[i * tmp_update_data_list_len:(i + 1) * tmp_update_data_list_len]
                                     for i in range(4)]
            tmp_update_data_lists.append(update_data_list[4 * tmp_update_data_list_len:])
            # 记录更新时间、索引体积、磁盘IO和查询时间
            logging.info("*************start %s************" % target_model_path)
            for j in range(5):
                logging.info("Update idx: %s" % j)
                start_time = time.time()
                index.insert(tmp_update_data_lists[j])
                index.save()
                end_time = time.time()
                logging.info("Update time: %s" % (end_time - start_time))
                structure_size, ie_size = index.size()
                logging.info("Structure size: %s" % structure_size)
                logging.info("Index entry size: %s" % ie_size)
                io_cost = index.io_cost
                logging.info("IO cost: %s" % io_cost)
                if index_info[0] is SLBRIN:
                    model_num = index.meta.last_hr + 1
                    logging.info("Model num: %s" % model_num)
                    model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
                    model_precisions_avg = sum(model_precisions) / model_num
                    logging.info("Error bound: %s" % model_precisions_avg)
                    logging.info("Sum up full cr time: %s" % index.sum_up_full_cr_time)
                    logging.info("Merge outdated cr time: %s" % index.merge_outdated_cr_time)
                    logging.info("Retrain inefficient model time: %s" % index.retrain_inefficient_model_time)
                    logging.info("Retrain inefficient model num: %s" % index.retrain_inefficient_model_num)
                    update_time = end_time - start_time - \
                                  index.sum_up_full_cr_time - index.merge_outdated_cr_time - index.retrain_inefficient_model_time
                    logging.info("Update time: %s" % update_time)
                # 查询跑5次，舍弃最慢和最快，剩余三个取平均，避免算力波动的影响
                search_time_list = [0] * 5
                search_io_cost_list = [0] * 5
                for k in range(0, 5):
                    start_time = time.time()
                    index.test_point_query(point_query_list)
                    end_time = time.time()
                    search_time = (end_time - start_time) / len(point_query_list)
                    search_io_cost = (index.io_cost - io_cost) / len(point_query_list)
                    io_cost = index.io_cost
                    search_time_list[k] = search_time
                    search_io_cost_list[k] = search_io_cost
                search_time_list = sorted(search_time_list)
                search_io_cost_list = sorted(search_io_cost_list)
                logging.info("Point query time: %s" % (sum(search_time_list[1:4]) / 3))
                logging.info("Point query io cost: %s" % (sum(search_io_cost_list[1:4]) / 3))
                for k in range(0, 5):
                    start_time = time.time()
                    index.test_range_query(range_query_list)
                    end_time = time.time()
                    search_time = (end_time - start_time) / len(range_query_list)
                    search_io_cost = (index.io_cost - io_cost) / len(range_query_list)
                    io_cost = index.io_cost
                    search_time_list[k] = search_time
                    search_io_cost_list[k] = search_io_cost
                search_time_list = sorted(search_time_list)
                search_io_cost_list = sorted(search_io_cost_list)
                logging.info("Range query time: %s" % (sum(search_time_list[1:4]) / 3))
                logging.info("Range query io cost: %s" % (sum(search_io_cost_list[1:4]) / 3))
                for k in range(0, 5):
                    start_time = time.time()
                    index.test_knn_query(knn_query_list)
                    end_time = time.time()
                    search_time = (end_time - start_time) / len(knn_query_list)
                    search_io_cost = (index.io_cost - io_cost) / len(knn_query_list)
                    io_cost = index.io_cost
                    search_time_list[k] = search_time
                    search_io_cost_list[k] = search_io_cost
                search_time_list = sorted(search_time_list)
                search_io_cost_list = sorted(search_io_cost_list)
                logging.info("KNN query time: %s" % (sum(search_time_list[1:4]) / 3))
                logging.info("KNN query io cost: %s" % (sum(search_io_cost_list[1:4]) / 3))
