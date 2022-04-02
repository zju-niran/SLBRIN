import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region
from src.spatial_index.geohash_model_index import GeoHashModelIndex

"""
1. 读取数据
2. 设置实验参数
3. 开始实验
3.1 快速构建精度低的
3.2 重新训练提高精度
"""
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/gm_index"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S %p")
    # 1. 读取数据
    # path = '../../data/trip_data_1_100000_sorted.npy'
    path = '../../data/trip_data_1_filter_sorted.npy'
    data_list = np.load(path).tolist()
    # 2. 设置实验参数
    n_list = [160000, 80000, 40000, 20000, 10000, 5000]
    # 3. 开始实验
    # 3.1 快速构建精度低的
    for n in n_list:
        model_path = "model/gm_index/n_" + str(n) + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        index = GeoHashModelIndex(model_path=model_path)
        index_name = index.name
        logging.info("*************start %s************" % model_path)
        start_time = time.time()
        index.build(data_list=data_list, threshold_number=n, data_precision=6, region=Region(40, 42, -75, -73),
                    use_threshold=False,
                    threshold=20,
                    core=[1, 128, 1],
                    train_step=500,
                    batch_size=1024,
                    learning_rate=0.01,
                    retrain_time_limit=20,
                    thread_pool_size=6,
                    save_nn=False)
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        index.save()
        model_num = index.sbrin.meta_page.size + 1
        logging.info("Model num: %s" % model_num)
        model_precisions = [(blk_range.model.max_err - blk_range.model.min_err)
                            for blk_range in index.sbrin.regular_pages if blk_range is not None]
        model_precisions_avg = sum(model_precisions) / model_num
        logging.info("Model precision avg: %s" % model_precisions_avg)
        path = '../../data/trip_data_1_point_query.csv'
        point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
        point_query_list = point_query_df.drop("count", axis=1).values.tolist()
        start_time = time.time()
        results = index.point_query(point_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(point_query_list)
        logging.info("Point query time: %s" % search_time)
    # 3.2 重新训练提高精度
    for n in n_list:
        model_path = "model/gm_index/n_" + str(n) + "/"
        index = GeoHashModelIndex(model_path=model_path)
        index_name = index.name
        logging.info("*************start %s************" % model_path)
        start_time = time.time()
        index.load()
        index.build_nn_multiprocess(use_threshold=True,
                                    threshold=20,
                                    core=[1, 128, 1],
                                    train_step=500,
                                    batch_size=1024,
                                    learning_rate=0.01,
                                    retrain_time_limit=20,
                                    thread_pool_size=6,
                                    save_nn=True)
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        index.save()
        model_num = index.sbrin.meta_page.size + 1
        logging.info("Model num: %s" % model_num)
        model_precisions = [(blk_range.model.max_err - blk_range.model.min_err)
                            for blk_range in index.sbrin.regular_pages if blk_range is not None]
        model_precisions_avg = sum(model_precisions) / model_num
        logging.info("Model precision avg: %s" % model_precisions_avg)
        path = '../../data/trip_data_1_point_query.csv'
        point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
        point_query_list = point_query_df.drop("count", axis=1).values.tolist()
        start_time = time.time()
        results = index.point_query(point_query_list)
        end_time = time.time()
        search_time = (end_time - start_time) / len(point_query_list)
        logging.info("Point query time: %s" % search_time)
