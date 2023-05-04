import logging
import os
import time

from src.experiment.common_utils import Distribution, load_data, load_query, data_region, data_precision
from src.si.brin_spatial import BRINSpatial
from src.si.kd_tree import KDTree
from src.si.pr_quad_tree import PRQuadTree
from src.si.r_tree import RTree
from src.proposed_sli.slibs import SLIBS
from src.sli.zm_index import ZMIndex

"""
实验探究：对比不同超参下RT/PRQT/KDT/BRINS/ZM/SLIBS的构建性能、存储成本和检索时间
"""
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/competitors/"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    index_infos = [
        # RT
        (RTree, "rtree/%s_0.4", 0.4),
        (RTree, "rtree/%s_0.5", 0.5),
        (RTree, "rtree/%s_0.6", 0.6),
        (RTree, "rtree/%s_0.7", 0.7),
        (RTree, "rtree/%s_0.8", 0.8),
        # PRQT
        (PRQuadTree, "prquadtree/%s_50", 50),
        (PRQuadTree, "prquadtree/%s_50", 100),
        (PRQuadTree, "prquadtree/%s_50", 250),
        (PRQuadTree, "prquadtree/%s_50", 500),
        # KDT
        (KDTree, "kdtree/%s", "kdtree/%s"),
        # BRINS
        (BRINSpatial, "brinspatial/%s_2", 2, False),
        (BRINSpatial, "brinspatial/%s_4", 4, False),
        (BRINSpatial, "brinspatial/%s_8", 8, False),
        (BRINSpatial, "brinspatial/%s_8_sorted", 8, True),
        (BRINSpatial, "brinspatial/%s_16_sorted", 16, True),
        (BRINSpatial, "brinspatial/%s_32_sorted", 32, True),
        (BRINSpatial, "brinspatial/%s_64_sorted", 64, True),
        (BRINSpatial, "brinspatial/%s_128_sorted", 128, True),
        (BRINSpatial, "brinspatial/%s_256_sorted", 256, True),
        # ZM
        (ZMIndex, "zmindex/%s_500", 500),
        (ZMIndex, "zmindex/%s_1000", 1000),
        (ZMIndex, "zmindex/%s_2000", 2000),
        (ZMIndex, "zmindex/%s_5000", 5000),
        (ZMIndex, "zmindex/%s_10000", 10000),
        # SLIBS
        (SLIBS, "slibs/%s_500", 500),
        (SLIBS, "slibs/%s_1000", 1000),
        (SLIBS, "slibs/%s_2000", 2000),
        (SLIBS, "slibs/%s_5000", 5000),
        (SLIBS, "slibs/%s_10000", 10000),
    ]
    for data_distribution in data_distributions:
        build_data_list = load_data(data_distribution, 0)
        point_query_list = load_query(data_distribution, 0).tolist()
        range_query_list = load_query(data_distribution, 1).tolist()
        knn_query_list = load_query(data_distribution, 2).tolist()
        for index_info in index_infos:
            model_path = parent_path + index_info[1] % data_distribution.name
            logging.info("*************start %s************" % model_path)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            start_time = time.time()
            if index_info[0] is RTree:
                index = RTree(model_path=model_path)
                index.build(data_list=build_data_list,
                            fill_factor=index_info[2],
                            leaf_node_capacity=113,
                            non_leaf_node_capacity=113,
                            buffering_capacity=0)
            elif index_info[0] is BRINSpatial:
                index = BRINSpatial(model_path=model_path)
                index.build(data_list=build_data_list,
                            pages_per_range=index_info[2],
                            is_sorted=index_info[3],
                            region=data_region[data_distribution],
                            data_precision=data_precision[data_distribution])
            elif index_info[0] is PRQuadTree:
                index = PRQuadTree(model_path=model_path)
                index.build(data_list=build_data_list,
                            region=data_region[data_distribution],
                            threshold_number=index_info[2],
                            data_precision=data_precision[data_distribution])
            elif index_info[0] is KDTree:
                index = KDTree(model_path=model_path)
                index.build(data_list=build_data_list)
            elif index_info[0] is ZMIndex:
                index = ZMIndex(model_path=model_path)
                index.build(data_list=build_data_list,
                            is_sorted=True,
                            data_precision=data_precision[data_distribution],
                            region=data_region[data_distribution],
                            is_new=True,
                            is_simple=False,
                            weight=1,
                            stages=[1, index_info[2]],
                            cores=[[1, 128], [1, 128]],
                            train_steps=[5000, 5000],
                            batch_nums=[64, 64],
                            learning_rates=[0.1, 0.1],
                            use_thresholds=[False, True],
                            thresholds=[0, 0],
                            retrain_time_limits=[5, 2],
                            thread_pool_size=8)
                stage1_model_precision = index.rmi[0][0].model.max_err - index.rmi[0][0].model.min_err
                logging.info("Stage1 model error bound: %s" % stage1_model_precision)
                stage2_model_precisions = [(node.model.max_err - node.model.min_err)
                                           for node in index.rmi[1] if node.model is not None]
                stage2_model_num = len(stage2_model_precisions)
                stage2_model_precisions_avg = sum(stage2_model_precisions) / stage2_model_num
                logging.info("Stage2 error bound: %s" % stage2_model_precisions_avg)
                logging.info("Stage2 model number: %s" % stage2_model_num)
            elif index_info[0] is SLIBS:
                index = SLIBS(model_path=model_path)
                index.build(data_list=build_data_list,
                            is_sorted=True,
                            data_precision=data_precision[data_distribution],
                            region=data_region[data_distribution],
                            is_new=True,
                            is_simple=False,
                            weight=1,
                            stages=[1, index_info[2]],
                            cores=[[1, 128], [1, 128]],
                            train_steps=[5000, 5000],
                            batch_nums=[64, 64],
                            learning_rates=[0.1, 0.1],
                            use_thresholds=[False, True],
                            thresholds=[0, 0],
                            retrain_time_limits=[5, 2],
                            thread_pool_size=8)
                stage1_model_precision = index.rmi[0][0].model.max_err - index.rmi[0][0].model.min_err
                logging.info("Stage1 model error bound: %s" % stage1_model_precision)
                stage2_model_precisions = [(node.model.max_err - node.model.min_err)
                                           for node in index.rmi[1] if node.model is not None]
                stage2_model_num = len(stage2_model_precisions)
                stage2_model_precisions_avg = sum(stage2_model_precisions) / stage2_model_num
                logging.info("Stage2 error bound: %s" % stage2_model_precisions_avg)
                logging.info("Stage2 model number: %s" % stage2_model_num)
            else:
                raise RuntimeError("Illegal index class")
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            io_cost = index.io_cost
            logging.info("IO cost: %s" % io_cost)
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
            for i in range(len(range_query_list) // 1000):
                tmp_range_query_list = range_query_list[i * 1000:(i + 1) * 1000]
                start_time = time.time()
                index.test_range_query(tmp_range_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / len(tmp_range_query_list)
                logging.info("Range query time: %s" % search_time)
                logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(tmp_range_query_list)))
                io_cost = index.io_cost
            for i in range(len(knn_query_list) // 1000):
                tmp_knn_query_list = knn_query_list[i * 1000:(i + 1) * 1000]
                start_time = time.time()
                index.test_knn_query(tmp_knn_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / len(tmp_knn_query_list)
                logging.info("KNN query time: %s" % search_time)
                logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(tmp_knn_query_list)))
                io_cost = index.io_cost
