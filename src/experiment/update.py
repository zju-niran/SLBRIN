import logging
import os
import shutil
import sys
import time

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.r_tree import RTree
from src.spatial_index.pr_quad_tree import PRQuadTree
from src.spatial_index.kd_tree import KDTree
from src.spatial_index.brin_spatial import BRINSpatial
from src.experiment.common_utils import Distribution, load_data, load_query, copy_dirs

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/update"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    indices = [RTree,
               PRQuadTree,
               KDTree,
               BRINSpatial,
               # ZMIndex,
               # SBRIN
               ]
    origin_model_paths = ["model/rtree/%s/fill_factor_0.8",
                          "model/prquadtree/%s/n_500",
                          "model/kdtree/%s",
                          "model/brinspatial/%s/ppr_64",
                          # "model/zmindex/%s/stage2_num_500",
                          # "model/sbrin/%s/tn_10000"
                          ]
    target_model_paths = ["model/update/rtree/%s",
                          "model/update/prquadtree/%s",
                          "model/update/kdtree/%s",
                          "model/update/brinspatial/%s",
                          # "model/update/zmindex/%s",
                          # "model/update/sbrin/%s"
                          ]
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    for data_distribution in data_distributions:
        for i in range(len(indices)):
            # 拷贝目标索引磁盘文件
            origin_model_path = origin_model_paths[i] % data_distribution.name
            target_model_path = target_model_paths[i] % data_distribution.name
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            copy_dirs(origin_model_path, target_model_path)
            # 加载索引结构到内存
            index = indices[i](model_path=target_model_path)
            index.load()
            # 更新数据集一分为五
            update_data_list = load_data(data_distribution, 1)
            update_data_len = len(update_data_list)
            tmp_update_data_list_len = update_data_len // 5 + 1
            tmp_update_data_lists = [update_data_list[i * tmp_update_data_list_len:(i + 1) * tmp_update_data_list_len]
                                     for i in range(4)]
            tmp_update_data_lists.append(update_data_list[4 * tmp_update_data_list_len:])
            # 记录更新时间、索引体积、磁盘IO和查询时间
            logging.info("*************start %s************" % target_model_path)
            for i in range(5):
                logging.info("Update idx: %s" % i)
                start_time = time.time()
                index.insert(tmp_update_data_lists[i])
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
