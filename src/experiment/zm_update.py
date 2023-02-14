import logging
import os
import shutil
import sys
import time

sys.path.append('/home/zju/wlj/SLBRIN')
from src.spatial_index.zm_index import ZMIndex
from src.experiment.common_utils import load_data, Distribution, copy_dirs, data_region, data_precision, \
    test_query

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    origin_path = "model/"
    target_path = "model/update/"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logging.basicConfig(filename=os.path.join(target_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    indices = [
        ZMIndex,
    ]
    origin_model_paths = [
        "zmindex/%s",
    ]
    target_model_paths = [
        "zmindex/%s"
    ]
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    # data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    for i in range(len(indices)):
        for data_distribution in data_distributions:
            origin_model_path = origin_path + origin_model_paths[i] % data_distribution.name
            target_model_path = target_path + target_model_paths[i] % data_distribution.name
            # index = indices[i](model_path=origin_model_path)
            # build_data_list = load_data(data_distribution, 0)
            # index.build(data_list=build_data_list,
            #             is_sorted=True,
            #             data_precision=data_precision[data_distribution],
            #             region=data_region[data_distribution],
            #             is_new=True,
            #             is_simple=False,
            #             weight=1,
            #             stages=[1, 100],
            #             cores=[[1, 32], [1, 32]],
            #             train_steps=[5000, 5000],
            #             batch_nums=[64, 64],
            #             learning_rates=[0.001, 0.001],
            #             use_thresholds=[True, False],
            #             thresholds=[5, 20],
            #             retrain_time_limits=[4, 2],
            #             thread_pool_size=6)
            # index.save()
            # 拷贝目标索引磁盘文件
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            copy_dirs(origin_model_path, target_model_path)
            # 加载索引结构到内存
            index = indices[i](model_path=target_model_path)
            index.load()
            # 更新数据集一分为五
            update_data_list = load_data(data_distribution, 1)
            update_data_len = len(update_data_list)
            tmp_update_data_list_len = update_data_len // 5 + 1
            tmp_update_data_lists = [update_data_list[i * tmp_update_data_list_len:(i + 1) * tmp_update_data_list_len]
                                     for i in range(4)]
            tmp_update_data_lists.append(update_data_list[4 * tmp_update_data_list_len:])
            # 记录更新时间、索引体积、磁盘IO和查询时间
            logging.info("*************start %s************" % target_model_path)
            logging.info("======Before update")
            logging.info("Structure size: %s" % index.size()[0])
            logging.info("Index entry size: %s" % index.size()[1])
            logging.info("Model precision avg: %s" % index.model_err())
            search_time, io_cost = test_query(index, data_distribution, 0)
            logging.info("Point query time: %s" % search_time)
            logging.info("Point query io cost: %s" % io_cost)
            search_time, io_cost = test_query(index, data_distribution, 1)
            logging.info("Range query time: %s" % search_time)
            logging.info("Range query io cost: %s" % io_cost)
            search_time, io_cost = test_query(index, data_distribution, 2)
            logging.info("KNN query time: %s" % search_time)
            logging.info("KNN query io cost: %s" % io_cost)
            for j in range(5):
                logging.info("======Update idx: %s" % j)
                start_time = time.time()
                index.insert(tmp_update_data_lists[j])
                end_time = time.time()
                logging.info("Update time: %s" % (end_time - start_time))
                start_time = time.time()
                index.update()
                index.save()
                end_time = time.time()
                logging.info("Update time: %s" % (end_time - start_time))
                structure_size, ie_size = index.size()
                logging.info("Structure size: %s" % structure_size)
                logging.info("Index entry size: %s" % ie_size)
                logging.info("Model precision avg: %s" % index.model_err())
                search_time, io_cost = test_query(index, data_distribution, 0)
                logging.info("Point query time: %s" % search_time)
                logging.info("Point query io cost: %s" % io_cost)
                search_time, io_cost = test_query(index, data_distribution, 1)
                logging.info("Range query time: %s" % search_time)
                logging.info("Range query io cost: %s" % io_cost)
                search_time, io_cost = test_query(index, data_distribution, 2)
                logging.info("KNN query time: %s" % search_time)
                logging.info("KNN query io cost: %s" % io_cost)
