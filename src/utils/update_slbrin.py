import logging
import os
import sys

sys.path.append('/home/zju/wlj/SLBRIN')
from src.experiment.common_utils import Distribution, data_precision, data_region, load_data
from src.spatial_index.slbrin import SLBRIN


def train_tn():
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    for data_distribution in data_distributions:
        model_path = "E:/model/update/slbrin/%s" % data_distribution.name
        index = SLBRIN(model_path=model_path)
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list,
                    is_sorted=True,
                    threshold_number=10000,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution],
                    threshold_err=1,
                    threshold_summary=1000,
                    threshold_merge=5,
                    is_new=False,
                    is_simple=False,
                    weight=1,
                    core=[1, 128],
                    train_step=5000,
                    batch_num=64,
                    learning_rate=0.1,
                    use_threshold=False,
                    threshold=0,
                    retrain_time_limit=0,
                    thread_pool_size=8)
        index.save()
        index.model_clear()
        structure_size, ie_size = index.size()
        logging.info("Structure size: %s" % structure_size)
        logging.info("Index entry size: %s" % ie_size)
        model_num = index.meta.last_hr + 1
        logging.info("Model num: %s" % model_num)
        model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
        model_precisions_avg = sum(model_precisions) / model_num
        logging.info("Model precision avg: %s" % model_precisions_avg)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/test"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    train_tn()
