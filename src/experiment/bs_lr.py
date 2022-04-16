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
    parent_path = "model/sbrin/lr_bn"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S %p")
    # 1. 读取数据
    path = '../../data/trip_data_1_10w_sorted.npy'
    # path = '../../data/trip_data_1_filter_sorted.npy'
    data_list = np.load(path).tolist()
    # 2. 设置实验参数
    # lrs = [0.1, 0.01, 0.001, 0.0001]
    lrs = [0.1, 0.01]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # 3. 开始实验
    for lr in lrs:
        for bs in batch_sizes:
            model_path = "model/gm_index/bs_lr/%s_%s/" % (bs, lr)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            index = SBRIN(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            index.build(data_list=data_list,
                        block_size=100,
                        threshold_number=1000,
                        data_precision=6,region=Region(40, 42, -75, -73),
                        use_threshold=True,
                        threshold=20,
                        core=[1, 128, 1],
                        train_step=5000,
                        batch_size=bs,
                        learning_rate=lr,
                        retrain_time_limit=2,
                        thread_pool_size=6,
                        save_nn=True,
                        weight=0.1)
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            index.save()
            model_num = index.meta.first_tmp_br
            logging.info("Model num: %s" % model_num)
            model_precisions = [(blk_range.model.max_err - blk_range.model.min_err)
                                for blk_range in index.block_ranges if blk_range.model is not None]
            model_precisions_avg = sum(model_precisions) / model_num
            logging.info("Model precision avg: %s" % model_precisions_avg)
