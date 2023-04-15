import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/SLBRIN')
from src.experiment.common_utils import Distribution, load_data, data_region, data_precision, load_query
from src.spatial_index.slibs import SLIBS
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/slibs"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    # stage2_nums = [500, 1000, 2000, 5000, 10000]
    stage2_nums = [1000]
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    for data_distribution in data_distributions:
        for stage2_num in stage2_nums:
            model_path = parent_path + "/%s/stage2_num_%s" % (data_distribution.name, stage2_num)
            if os.path.exists(model_path) is False:
                os.makedirs(model_path)
            index = SLIBS(model_path=model_path)
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            build_data_list = load_data(data_distribution, 0)
            index.build(data_list=build_data_list,
                        is_sorted=True,
                        data_precision=data_precision[data_distribution],
                        region=data_region[data_distribution],
                        is_new=True,
                        is_simple=False,
                        weight=1,
                        stages=[1, stage2_num],
                        cores=[[1, 128], [1, 128]],
                        train_steps=[5000, 5000],
                        batch_nums=[64, 64],
                        learning_rates=[0.1, 0.1],
                        use_thresholds=[True, True],
                        thresholds=[0, 0],
                        retrain_time_limits=[3, 5],
                        thread_pool_size=2)
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            index.model_clear()
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            io_cost = index.io_cost
            logging.info("IO cost: %s" % io_cost)
            stage1_model_precision = index.rmi[0][0].model.max_err - index.rmi[0][0].model.min_err
            logging.info("Stage1 model precision: %s" % stage1_model_precision)
            stage2_model_precisions = [(node.model.max_err - node.model.min_err)
                                       for node in index.rmi[1] if node.model is not None]
            stage2_model_num = len(stage2_model_precisions)
            stage2_model_precisions_avg = sum(stage2_model_precisions) / stage2_model_num
            logging.info("Stage2 model precision avg: %s" % stage2_model_precisions_avg)
            logging.info("Stage2 model number: %s" % stage2_model_num)

