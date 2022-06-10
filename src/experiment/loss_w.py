import logging
import os
import sys
import time

sys.path.append('/home/zju/wlj/st-learned-index')
from src.experiment.common_utils import Distribution, load_data, data_region, data_precision
from src.spatial_index.sbrin import SBRIN

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/sbrin"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    weights = [0.01, 0.1, 1, 10, 100, 1000]
    for data_distribution in data_distributions:
        for weight in weights:
            model_path = "model/sbrin/%s/loss_w_%s" % (data_distribution.name, weight)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            index = SBRIN(model_path=model_path)
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            build_data_list = load_data(data_distribution, 0)
            index.build(data_list=build_data_list,
                        is_sorted=True,
                        threshold_number=20000,
                        data_precision=data_precision[data_distribution],
                        region=data_region[data_distribution],
                        threshold_err=0,
                        threshold_summary=0,
                        threshold_merge=0,
                        is_new=True,
                        is_simple=False,
                        is_gpu=True,
                        weight=weight,
                        core=[1, 128],
                        train_step=5000,
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
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            logging.info("IO cost: %s" % index.io())
            model_num = index.meta.last_hr + 1
            logging.info("Model num: %s" % model_num)
            model_precisions = [(hr.model.max_err - hr.model.min_err)
                                for hr in index.history_ranges if hr.model]
            model_precisions_avg = sum(model_precisions) / model_num
            logging.info("Model precision avg: %s" % model_precisions_avg)
