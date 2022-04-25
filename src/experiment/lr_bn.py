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
                        format="%(message)s")
    # data_path = '../../data/index/trip_data_1_filter_10w_sorted.npy'
    data_path = '../../data/index/trip_data_1_filter_sorted.npy'
    # lrs = [0.1, 0.01, 0.001, 0.0001]
    lrs = [0.1]
    batch_nums = [16, 32, 64, 128, 256, 512]
    for lr in lrs:
        for batch_num in batch_nums:
            model_path = "model/sbrin/lr_bn/%s_%s/" % (lr, batch_num)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            index = SBRIN(model_path=model_path)
            index_name = index.name
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            data_list = np.load(data_path, allow_pickle=True)
            index.build(data_list=data_list,
                        threshold_number=20000,
                        data_precision=6,
                        region=Region(40, 42, -75, -73),
                        use_threshold=True,
                        threshold=0,
                        core=[1, 128, 1],
                        train_step=5000,
                        batch_num=batch_num,
                        learning_rate=lr,
                        retrain_time_limit=0,
                        thread_pool_size=12,
                        save_nn=True,
                        weight=1)
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            logging.info("Index size: %s" % index.size())
            model_num = index.meta.first_tmp_blk
            logging.info("Model num: %s" % model_num)
            model_precisions = [(blk.model.max_err - blk.model.min_err)
                                for blk in index.block_ranges if blk.model]
            model_precisions_avg = sum(model_precisions) / model_num
            logging.info("Model precision avg: %s" % model_precisions_avg)
