import logging
import os
import shutil
import time

from src.experiment.common_utils import Distribution, load_query, load_data, copy_dirs, group_data_by_date
from src.proposed_sli.tsusli import TSUSLI
from src.proposed_sli.uslbrin import USLBRIN

"""
实验探究：对比启用或不启用tcrm下uslbrin/tsusli的整体性能
1. 误差阈值控制机制：不同误差阈值（tel、tef、ten）下，学习模型和预测模型的性能对比
2. 历史增量学习机制：以随机初始化权重和历史模型为起点，学习模型的性能变化
"""
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    origin_path = "model/"
    target_path = "model/compare_uslbrin/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.basicConfig(filename=os.path.join(target_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    index_infos = [
        # tel
        ("uslbrin_0.8_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 0.8, 1, 1.1, True),
        ("uslbrin_0.9_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 0.9, 1, 1.1, False),
        ("uslbrin_1.0_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 1.0, 1, 1.1, False),
        ("uslbrin_1.1_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 1.1, 1, 1.1, False),
        ("uslbrin_1.2_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 1.2, 1, 1.1, False),
        # tef
        ("uslbrin_1.1_0_1.1", False, -1, 6, False, True, -1, 6, False, False, 1.1, 0, 1.1, False),
        # ("uslbrin_1.1_1_1.1", False, -1, 6, False, True, -1, 6, False, False, 1.1, 1, 1.1, False),
        ("uslbrin_1.1_2_1.1", False, -1, 6, False, True, -1, 6, False, False, 1.1, 2, 1.1, False),
        ("uslbrin_1.1_4_1.1", False, -1, 6, False, True, -1, 6, False, False, 1.1, 4, 1.1, False),
        ("uslbrin_1.1_8_1.1", False, -1, 6, False, True, -1, 6, False, False, 1.1, 8, 1.1, False),
        # ten
        ("uslbrin_1.1_1_0.8", False, -1, 6, False, True, -1, 6, False, False, 1.1, 1, 0.8, False),
        ("uslbrin_1.1_1_0.9", False, -1, 6, False, True, -1, 6, False, False, 1.1, 1, 0.9, False),
        ("uslbrin_1.1_1_1.0", False, -1, 6, False, True, -1, 6, False, False, 1.1, 1, 1.0, False),
        # ("uslbrin_1.1_1_1.1", False, -1, 6, False, True, -1, 6, False, False, 1.1, 1, 1.1, False),
        ("uslbrin_1.1_1_1.2", False, -1, 6, False, True, -1, 6, False, False, 1.1, 1, 1.2, False),
        # rm
        ("tsusli_1.1_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 1.1, 1, 1.1, True),
        ("tsusli_1.1_1_1.1", True, -1, 6, True, True, -1, 6, False, True, 1.1, 1, 1.1, False),
        # ("uslbrin_1.1_1_1.1", True, -1, 6, True, True, -1, 6, False, False, 1.1, 1, 1.1, False),
        ("uslbrin_1.1_1_1.1", True, -1, 3, True, True, -1, 3, False, True, 1.1, 1, 1.1, False),
    ]
    data_distributions = [Distribution.NYCT_SORTED]
    # data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    for data_distribution in data_distributions:
        point_query_list = load_query(data_distribution, 0).tolist()
        update_data_list = load_data(data_distribution, 1)
        # 2013-02-01-08: 1359676800 | 2013-02-02-08: 1359763200 | 2013-02-08-08: 1360281600
        # update_data_list = filter_data_by_date(update_data_list, 1359763200)
        # update_data_list = group_data_by_date(update_data_list, 1359676800, 60 * 10)
        update_data_list = group_data_by_date(update_data_list, 1359676800, 60 * 60 * 4)
        for index_info in index_infos:
            if index_info[0].startswith("tsusli"):
                origin_model_path = origin_path + "slibs/" + data_distribution.name + "/stage2_num_1000"
                index_class = TSUSLI
            else:
                origin_model_path = origin_path + "slbrin/" + data_distribution.name + "/tn_10000"
                index_class = USLBRIN
            target_model_path = target_path + data_distribution.name + "/" + index_info[0]
            if index_info[-1]:
                # copy the zm_index as the basic
                if os.path.exists(target_model_path):
                    shutil.rmtree(target_model_path)
                copy_dirs(origin_model_path, target_model_path)
            # initial the compared model from the zm_index
            logging.info("*************start %s %s************" % (index_info[0], data_distribution.name))
            start_time = time.time()
            index = index_class(model_path=target_model_path)
            super(index_class, index).load()
            index.build_append(time_interval=60 * 60 * 24,
                               start_time=1356998400,
                               end_time=1359676799,
                               lag=7,
                               predict_step=7,
                               cdf_width=100,
                               child_length=1,
                               cdf_model='var',
                               max_key_model='es',
                               is_init=index_info[9],
                               threshold_err=index_info[10],
                               threshold_err_cdf=index_info[11],
                               threshold_err_max_key=index_info[12],
                               is_retrain=index_info[1],
                               time_retrain=index_info[2],
                               thread_retrain=index_info[3],
                               is_save=index_info[4],
                               is_retrain_delta=index_info[5],
                               time_retrain_delta=index_info[6],
                               thread_retrain_delta=index_info[7],
                               is_save_delta=index_info[8],
                               is_build=index_info[-1])
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            logging.info("Error bound: %s" % index.model_err())
            for update_data in update_data_list:
                index.insert(update_data)
                logging.info("Update data num: %s" % len(update_data))
                io_cost = index.io_cost
                start_time = time.time()
                index.test_point_query(point_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / len(point_query_list)
                logging.info("Point query time: %s" % search_time)
                logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
