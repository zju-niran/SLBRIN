import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/SLBRIN')
from src.spatial_index.slbrin import SLBRIN
from src.spatial_index.zm_index import ZMIndex
from src.experiment.common_utils import Distribution, load_data, data_precision, data_region

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/build/"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    for data_distribution in data_distributions:
        logging.info("*************start %s************" % data_distribution)
        build_data_list = load_data(data_distribution, 0)
        # ZM: 多进程 + GPU
        logging.info("*************start ZM multi processes and GPU************")
        start_time = time.time()
        index = ZMIndex(model_path=parent_path + data_distribution.name + "/zm_mp_gpu/")
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    is_new=True,
                    is_simple=True,
                    is_gpu=True,
                    weight=1,
                    stages=[1, 5000],
                    cores=[[1, 128], [1, 128]],
                    train_steps=[500, 500],
                    batch_nums=[64, 64],
                    learning_rates=[0.1, 0.1],
                    use_thresholds=[False, False],
                    thresholds=[0, 0],
                    retrain_time_limits=[0, 0],
                    thread_pool_size=12)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        # SLBRIN: 多进程 + GPU
        logging.info("*************start SLBRIN multi processes and GPU************")
        start_time = time.time()
        index = SLBRIN(model_path=parent_path + data_distribution.name + "/slbrin_mp_gpu/")
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
                    is_gpu=True,
                    weight=1,
                    core=[1, 128],
                    train_step=500,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=0,
                    thread_pool_size=12)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        # ZM: 单进程 + GPU
        logging.info("*************start ZM single processes and GPU************")
        start_time = time.time()
        index = ZMIndex(model_path=parent_path + data_distribution.name + "/zm_sp_gpu/")
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    is_new=True,
                    is_simple=True,
                    is_gpu=True,
                    weight=1,
                    stages=[1, 5000],
                    cores=[[1, 128], [1, 128]],
                    train_steps=[500, 500],
                    batch_nums=[64, 64],
                    learning_rates=[0.1, 0.1],
                    use_thresholds=[False, False],
                    thresholds=[0, 0],
                    retrain_time_limits=[0, 0],
                    thread_pool_size=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        # SLBRIN: 单进程 + GPU
        logging.info("*************start SLBRIN single processes and GPU************")
        start_time = time.time()
        index = SLBRIN(model_path=parent_path + data_distribution.name + "/slbrin_sp_gpu/")
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
                    is_gpu=True,
                    weight=1,
                    core=[1, 128],
                    train_step=500,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=0,
                    thread_pool_size=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        # ZM: 单进程 + CPU
        logging.info("*************start ZM single processes and CPU************")
        start_time = time.time()
        index = ZMIndex(model_path=parent_path + data_distribution.name + "/zm_sp_cpu/")
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    is_new=True,
                    is_simple=True,
                    is_gpu=False,
                    weight=1,
                    stages=[1, 5000],
                    cores=[[1, 128], [1, 128]],
                    train_steps=[500, 500],
                    batch_nums=[64, 64],
                    learning_rates=[0.1, 0.1],
                    use_thresholds=[False, False],
                    thresholds=[0, 0],
                    retrain_time_limits=[0, 0],
                    thread_pool_size=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
        # SLBRIN: 单进程 + CPU
        logging.info("*************start SLBRIN single processes and CPU************")
        start_time = time.time()
        index = SLBRIN(model_path=parent_path + data_distribution.name + "/slbrin_sp_cpu/")
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
                    is_gpu=False,
                    weight=1,
                    core=[1, 128],
                    train_step=500,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=0,
                    thread_pool_size=1)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time: %s" % build_time)
