import logging
import os
import shutil
import sys
import time

sys.path.append('/home/zju/wlj/SLBRIN')
from src.spatial_index.r_tree import RTree
from src.spatial_index.pr_quad_tree import PRQuadTree
from src.spatial_index.kd_tree import KDTree
from src.spatial_index.brin_spatial import BRINSpatial
from src.spatial_index.zm_index import ZMIndex
from src.spatial_index.slbrin import SLBRIN
from src.experiment.common_utils import Distribution, load_data, load_query, copy_dirs

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    origin_path = "model/"
    target_path = "model/update/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.basicConfig(filename=os.path.join(target_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    indices = [RTree,
               PRQuadTree,
               KDTree,
               BRINSpatial,
               ZMIndex,
               SLBRIN
               ]
    origin_model_paths = ["rtree/%s/fill_factor_0.8",
                          "prquadtree/%s/n_500",
                          "kdtree/%s",
                          "brinspatial/%s_SORTED/ppr_sorted_64",
                          "zmindex/%s_SORTED/stage2_num_5000",
                          "slbrin/%s_SORTED/tn_10000"
                          ]
    target_model_paths = ["rtree/%s",
                          "prquadtree/%s",
                          "kdtree/%s",
                          "brinspatial/%s",
                          "zmindex/%s",
                          "slbrin/%s"
                          ]
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    for i in range(len(indices)):
        for data_distribution in data_distributions:
            # 拷贝目标索引磁盘文件
            origin_model_path = origin_path + origin_model_paths[i] % data_distribution.name
            target_model_path = target_path + target_model_paths[i] % data_distribution.name
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            copy_dirs(origin_model_path, target_model_path)
            # 加载索引结构到内存
            index = indices[i](model_path=target_model_path)
            index.load()
            if index.name == 'SLBRIN':
                index.meta.threshold_err = 1.0
                index.is_gpu = False
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
                logging.info("IO cost: %s" % index.io())
                if index.name == "SLBRIN":
                    model_num = index.meta.last_hr + 1
                    logging.info("Model num: %s" % model_num)
                    model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
                    model_precisions_avg = sum(model_precisions) / model_num
                    logging.info("Model precision avg: %s" % model_precisions_avg)
                    logging.info("Sum up full cr time: %s" % index.sum_up_full_cr_time)
                    logging.info("Merge outdated cr time: %s" % index.merge_outdated_cr_time)
                    logging.info("Retrain inefficient model time: %s" % index.retrain_inefficient_model_time)
                    logging.info("Retrain inefficient model num: %s" % index.retrain_inefficient_model_num)
                    update_time = end_time - start_time - \
                                  index.sum_up_full_cr_time - index.merge_outdated_cr_time - index.retrain_inefficient_model_time
                    logging.info("Update time: %s" % update_time)
                # 查询跑多次，避免算力波动的影响
                for k in range(5):
                    point_query_list = load_query(data_distribution, 0).tolist()
                    start_time = time.time()
                    index.test_point_query(point_query_list)
                    end_time = time.time()
                    search_time = (end_time - start_time) / len(point_query_list)
                    logging.info("Point query time: %s" % search_time)
                for k in range(5):
                    range_query_list = load_query(data_distribution, 1).tolist()
                    start_time = time.time()
                    index.test_range_query(range_query_list)
                    end_time = time.time()
                    search_time = (end_time - start_time) / len(range_query_list)
                    logging.info("Range query time: %s" % search_time)
                for k in range(5):
                    knn_query_list = load_query(data_distribution, 2).tolist()
                    start_time = time.time()
                    index.test_knn_query(knn_query_list)
                    end_time = time.time()
                    search_time = (end_time - start_time) / len(knn_query_list)
                    logging.info("KNN query time: %s" % search_time)
