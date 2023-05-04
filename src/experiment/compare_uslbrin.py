import logging
import os
import shutil
import time

from src.experiment.common_utils import Distribution, load_data, load_query, copy_dirs, group_data_by_date
from src.spatial_index.brin_spatial import BRINSpatial
from src.spatial_index.pr_quad_tree import PRQuadTree
from src.spatial_index.r_tree import RTree
from src.spatial_index.tsusli import TSUSLI
from src.spatial_index.uslbrin import USLBRIN

"""
实验探究：对比RT/PRQT/BRINS/TSUSLI/USLBRIN的整体性能
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
        (BRINSpatial, "brinspatial/%s_SORTED_sorted_64", "brinspatial/%s"),
        (TSUSLI, "slibs/%s_SORTED_1000", "tsusli/%s"),
        (USLBRIN, "slbrin/%s_SORTED_10000", "uslbrin/%s"),
    ]
    data_distributions = [Distribution.NYCT_SORTED, Distribution.NORMAL_SORTED, Distribution.UNIFORM_SORTED]
    for data_distribution in data_distributions:
        point_query_list = load_query(data_distribution, 0).tolist()
        range_query_list = load_query(data_distribution, 1).tolist()[2000:3000]
        knn_query_list = load_query(data_distribution, 2).tolist()[2000:3000]
        update_data_list = load_data(data_distribution, 1)
        update_data_list = group_data_by_date(update_data_list, 1359676800, 60 * 60 * 4)
        for index_info in index_infos:
            # 拷贝目标索引磁盘文件
            origin_model_path = origin_path + index_info[1] % data_distribution.name
            target_model_path = target_path + index_info[2] % data_distribution.name
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            copy_dirs(origin_model_path, target_model_path)
            # 加载索引结构到内存
            if index_info[0] in [TSUSLI, USLBRIN]:
                index = index_info[0](model_path=target_model_path)
                super(index_info[0], index).load()
                index.build_append(time_interval=60 * 60 * 24,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   lag=7,
                                   predict_step=7,
                                   cdf_width=100,
                                   child_length=1,
                                   cdf_model='var',
                                   max_key_model='es',
                                   is_init=True,
                                   threshold_err=1.1,
                                   threshold_err_cdf=1,
                                   threshold_err_max_key=1.1,
                                   is_retrain=True,
                                   time_retrain=-1,
                                   thread_retrain=6,
                                   is_save=False,
                                   is_retrain_delta=True,
                                   time_retrain_delta=-1,
                                   thread_retrain_delta=6,
                                   is_save_delta=False,
                                   is_build=True)
            else:
                index = index_info[0](model_path=target_model_path)
                index.load()
            logging.info("*************start %s************" % target_model_path)
            for j in range(len(update_data_list)):
                logging.info("Update idx: %s" % j)
                logging.info("Update data num: %s" % len(update_data_list[j]))
                io_cost = index.io_cost
                start_time = time.time()
                index.insert(update_data_list[j])
                index.save()
                end_time = time.time()
                logging.info("Update time: %s" % (end_time - start_time))
                logging.info("Update io cost: %s" % (index.io_cost - io_cost))
                io_cost = index.io_cost
                structure_size, ie_size = index.size()
                logging.info("Structure size: %s" % structure_size)
                logging.info("Index entry size: %s" % ie_size)
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
