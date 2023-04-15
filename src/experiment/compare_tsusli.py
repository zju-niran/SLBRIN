import logging
import os
import shutil
import sys
import time

sys.path.append('/home/zju/wlj/SLBRIN')
from src.spatial_index.tsusli import TSUSLI
from src.spatial_index.dtusli import ZMIndexDeltaInsert
from src.spatial_index.ipusli import ZMIndexInPlaceInsert
from src.experiment.common_utils import load_data, Distribution, copy_dirs, load_query, filter_data_by_date, \
    group_data_by_date

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    origin_path = "model/origin/"
    target_path = "model/compare_tsusli/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.basicConfig(filename=os.path.join(target_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    index_infos = [
        # prepare
        # ("dtusli", True, -1, 6, True, 146, False, 0),
        # predict_step
        # ("tsusli_24_1_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 1, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_6_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 6, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_18_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 18, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_24_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 24, 100, 1, "var", "es", False, 0, False),
        # bs
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_146_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 146, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_292_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 292, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_438_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 438, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_584_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 584, "var", "es", False, 0, False),
        # cdf_width
        # ("tsusli_24_12_10_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 10, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_50_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 50, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_250_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 250, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_500_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 500, 1, "var", "es", False, 0, False),
        # ts model
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, True, -1, 6, True, 24, 12, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_1_var_sarima", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "sarima", False, 0, False),
        # ("tsusli_24_12_100_1_var_rnn", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "rnn", False, 0, False),
        # ("tsusli_24_12_100_1_var_lstm", False, -1, 6, False, True, -1, 6, True, 24, 12, 100, 1, "var", "lstm", True),
        # ("tsusli_24_12_100_1_var_gru", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "gru", False, 0, False),
        # sts model
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, True, -1, 6, True, 24, 12, 100, 1, "var", "es", False, 0, False),
        # ("tsusli_24_12_100_1_fclstm_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "fclstm", "es", False, 0, False),
        # ("tsusli_24_12_100_1_convlstm_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "convlstm", "es", False, 0, False),
        # compare update among dtusli/ipusli/tsusli
        # ("dtusli", False, -1, 6, False, 146, False, 0),
        # ("ipusli", False, -1, 6, False, 0.2, False, 0),
        # ("ipusli2", False, -1, 6, False, 0.8, False, 0),
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "es", False, 0, False),
        # compre retrain among dtusli/ipusli/tsusli
        # ("dtusli", False, -1, 6, False, 146, True, 0),
        # ("dtusli", False, -1, 6, False, 146, True, 1),
        # ("dtusli", False, -1, 6, False, 146, True, 2),
        # ("dtusli", False, -1, 6, False, 146, True, 4),
        # ("dtusli", False, -1, 6, False, 146, True, 8),
        # ("ipusli", False, -1, 6, False, 0.2, True, 0),
        # ("ipusli2", False, -1, 6, False, 0.8, True, 0),
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "es", False, 0, False),
        # ("dtusli", False, -1, 6, False, 146, True, 0),
        # ("ipusli", False, -1, 6, False, 0.2, True, 0),
        # ("ipusli2", False, -1, 6, False, 0.8, True, 0),
        # ("tsusli_24_12_100_1_var_es", False, -1, 6, False, False, -1, 6, False, 24, 12, 100, 1, "var", "es", False, 0, False),
    ]
    data_distributions = [Distribution.NYCT_SORTED]
    # data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    for data_distribution in data_distributions:
        origin_model_path = origin_path + data_distribution.name
        point_query_list = load_query(data_distribution, 0).tolist()
        update_data_list = load_data(data_distribution, 1)
        # 2013-02-01-08: 1359676800 | 2013-02-02-08: 1359763200 | 2013-02-08-08: 1360281600
        update_data_list = filter_data_by_date(update_data_list, 1359763200)
        update_data_list = group_data_by_date(update_data_list, 1359676800, 60 * 10)
        for index_info in index_infos:
            target_model_path = target_path + data_distribution.name + "/" + index_info[0]
            if index_info[-1]:
                # copy the zm_index as the basic
                if os.path.exists(target_model_path):
                    shutil.rmtree(target_model_path)
                copy_dirs(origin_model_path, target_model_path)
            # initial the compared model from the zm_index
            logging.info("*************start %s %s************" % (index_info[0], data_distribution.name))
            start_time = time.time()
            if index_info[0].startswith("dtusli"):
                index = ZMIndexDeltaInsert(model_path=target_model_path)
                super(ZMIndexDeltaInsert, index).load()
                index.build_append(time_interval=60 * 60,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   initial_length=index_info[5],
                                   is_init=index_info[6],
                                   threshold_err=index_info[7],
                                   is_retrain=index_info[1],
                                   time_retrain=index_info[2],
                                   thread_retrain=index_info[3],
                                   is_save=index_info[4])
            elif index_info[0].startswith("tsusli"):
                index = TSUSLI(model_path=target_model_path)
                super(TSUSLI, index).load()
                index.build_append(time_interval=60 * 60,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   lag=index_info[9],
                                   predict_step=index_info[10],
                                   cdf_width=index_info[11],
                                   child_length=index_info[12],
                                   cdf_model=index_info[13],
                                   max_key_model=index_info[14],
                                   is_init=index_info[15],
                                   threshold_err=index_info[16],
                                   threshold_err_cdf=index_info[17],
                                   threshold_err_max_key=index_info[18],
                                   is_retrain=index_info[1],
                                   time_retrain=index_info[2],
                                   thread_retrain=index_info[3],
                                   is_save=index_info[4],
                                   is_retrain_delta=index_info[5],
                                   time_retrain_delta=index_info[6],
                                   thread_retrain_delta=index_info[7],
                                   is_save_delta=index_info[8],
                                   is_build=index_info[-1])
            else:
                index = ZMIndexInPlaceInsert(model_path=target_model_path)
                super(ZMIndexInPlaceInsert, index).load()
                index.build_append(time_interval=60 * 60,
                                   start_time=1356998400,
                                   end_time=1359676799,
                                   empty_ratio=index_info[5],
                                   is_init=index_info[6],
                                   threshold_err=index_info[7],
                                   is_retrain=index_info[1],
                                   time_retrain=index_info[2],
                                   thread_retrain=index_info[3],
                                   is_save=index_info[4])
            index.save()
            end_time = time.time()
            build_time = end_time - start_time
            logging.info("Build time: %s" % build_time)
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            logging.info("Model precision avg: %s" % index.model_err())
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
