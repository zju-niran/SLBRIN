import logging
import os
import time

import tensorflow as tf

from src.experiment.common_utils import Distribution, load_data, data_precision, data_region
from src.proposed_sli.slbrin import SLBRIN
from src.sli.zm_index import ZMIndex

"""
实验探究：对比在多线程和GPU下ZM/SLIBS/SLBRIN的并发度
"""
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/build/"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    index_infos = [
        (ZMIndex, "zmindex/%s_12_gpu", 12, True),
        (SLBRIN, "slbrin/%s_12_gpu", 12, True),
        (ZMIndex, "zmindex/%s_1_gpu", 1, True),
        (SLBRIN, "slbrin/%s_1_gpu", 1, True),
        (ZMIndex, "zmindex/%s_1_cpu", 1, False),
        (SLBRIN, "slbrin/%s_1_cpu", 1, False)]
    for data_distribution in data_distributions:
        build_data_list = load_data(data_distribution, 0)
        for index_info in index_infos:
            model_path = parent_path + index_info[1] % data_distribution.name
            logging.info("*************start %s %s************" % model_path)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            if index_info[3]:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            start_time = time.time()
            if index_info[0] is ZMIndex:
                index = ZMIndex(model_path=model_path)
                index.build(data_list=build_data_list,
                            is_sorted=True,
                            data_precision=data_precision[data_distribution],
                            region=data_region[data_distribution],
                            is_new=True,
                            is_simple=True,
                            weight=1,
                            stages=[1, 5000],
                            cores=[[1, 128], [1, 128]],
                            train_steps=[500, 500],
                            batch_nums=[64, 64],
                            learning_rates=[0.1, 0.1],
                            use_thresholds=[False, False],
                            thresholds=[0, 0],
                            retrain_time_limits=[0, 0],
                            thread_pool_size=index_info[2])
            elif index_info[0] is SLBRIN:
                index = SLBRIN(model_path=model_path)
                index.build(data_list=build_data_list,
                            is_sorted=True,
                            threshold_number=10000,
                            data_precision=data_precision[data_distribution],
                            region=data_region[data_distribution],
                            threshold_err=0,
                            threshold_summary=0,
                            threshold_merge=0,
                            is_new=True,
                            is_simple=True,
                            weight=1,
                            core=[1, 128],
                            train_step=500,
                            batch_num=64,
                            learning_rate=0.1,
                            use_threshold=False,
                            threshold=0,
                            retrain_time_limit=0,
                            thread_pool_size=index_info[2])
            else:
                raise RuntimeError("Illegal index class")
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
