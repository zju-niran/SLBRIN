import logging
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.common_utils import Region
from src.spatial_index.sbrin import SBRIN

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/sbrin/loss_w"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S %p")
    # 1. 读取数据
    data_path = '../../data/index/trip_data_1_filter_10w_sorted.npy'
    # data_path = '../../data/index/trip_data_1_filter_sorted.npy'
    # 2. 设置实验参数
    weights = [0.01, 0.1, 1, 10, 100, 1000]
    # 3. 开始实验
    for weight in weights:
        model_path = "model/sbrin/loss_w/w_%s/" % weight
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        index = SBRIN(model_path=model_path)
        index_name = index.name
        logging.info("*************start %s************" % model_path)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)
        index.build(data_list=data_list,
                    block_size=100,
                    threshold_number=1000,
                    data_precision=6,
                    region=Region(40, 42, -75, -73),
                    use_threshold=True,
                    threshold=0,
                    core=[1, 128, 1],
                    train_step=5000,
                    batch_num=64,
                    learning_rate=0.1,
                    retrain_time_limit=4,
                    thread_pool_size=10,
                    save_nn=True,
                    weight=weight)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        model_num = index.meta.first_tmp_br
        logging.info("Model num: %s" % model_num)
        model_precisions = [(blk_range.model.max_err - blk_range.model.min_err)
                            for blk_range in index.block_ranges if blk_range.model is not None]
        model_precisions_avg = sum(model_precisions) / model_num
        logging.info("Model precision avg: %s" % model_precisions_avg)
