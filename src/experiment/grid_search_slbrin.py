import logging
import os
import shutil
import time

from src.experiment.common_utils import Distribution, load_query, load_data, data_precision, data_region, copy_dirs
from src.proposed_sli.slbrin import SLBRIN


def grid_search_w_lr_bs():
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    weights = [0.01, 0.1, 1, 10, 100, 1000]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    batch_nums = [16, 32, 64, 128, 256, 512]
    for data_distribution in data_distributions:
        for weight in weights:
            for lr in lrs:
                for batch_num in batch_nums:
                    model_path = "model/slbrin/%s_%s_%s_%s" % (data_distribution.name, weight, lr, batch_num)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    index = SLBRIN(model_path=model_path)
                    logging.info("*************start %s************" % model_path)
                    start_time = time.time()
                    build_data_list = load_data(data_distribution, 0)
                    index.build(data_list=build_data_list,
                                is_sorted=True,
                                threshold_number=10000,
                                data_precision=data_precision[data_distribution],
                                region=data_region[data_distribution],
                                threshold_err=1,
                                threshold_summary=1000,
                                threshold_merge=5,
                                is_new=True,
                                is_simple=False,
                                weight=weight,
                                core=[1, 128],
                                train_step=5000,
                                batch_num=batch_num,
                                learning_rate=lr,
                                use_threshold=False,
                                threshold=0,
                                retrain_time_limit=0,
                                thread_pool_size=12)
                    index.save()
                    end_time = time.time()
                    build_time = end_time - start_time
                    logging.info("Build time: %s" % build_time)
                    logging.info("Build io cost: %s" % index.io_cost)
                    io_cost = index.io_cost
                    index.model_clear()
                    structure_size, ie_size = index.size()
                    logging.info("Structure size: %s" % structure_size)
                    logging.info("Index entry size: %s" % ie_size)
                    model_num = index.meta.last_hr + 1
                    logging.info("Model num: %s" % model_num)
                    model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
                    model_precisions_avg = sum(model_precisions) / model_num
                    logging.info("Error bound: %s" % model_precisions_avg)


def grid_search_tn():
    data_distributions = [Distribution.UNIFORM_SORTED, Distribution.NORMAL_SORTED, Distribution.NYCT_SORTED]
    tn_list = [5000, 6000, 8000, 10000, 20000]
    for data_distribution in data_distributions:
        for tn in tn_list:
            model_path = "model/slbrin/%s_%s" % (data_distribution.name, tn)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            index = SLBRIN(model_path=model_path)
            logging.info("*************start %s************" % model_path)
            start_time = time.time()
            build_data_list = load_data(data_distribution, 0)
            index.build(data_list=build_data_list,
                        is_sorted=True,
                        threshold_number=tn,
                        data_precision=data_precision[data_distribution],
                        region=data_region[data_distribution],
                        threshold_err=1,
                        threshold_summary=1000,
                        threshold_merge=5,
                        is_new=True,
                        is_simple=False,
                        weight=1,
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
            logging.info("Build io cost: %s" % index.io_cost)
            io_cost = index.io_cost
            index.model_clear()
            structure_size, ie_size = index.size()
            logging.info("Structure size: %s" % structure_size)
            logging.info("Index entry size: %s" % ie_size)
            model_num = index.meta.last_hr + 1
            logging.info("Model num: %s" % model_num)
            model_precisions = [(hr.model.max_err - hr.model.min_err) for hr in index.history_ranges]
            model_precisions_avg = sum(model_precisions) / model_num
            logging.info("Error bound: %s" % model_precisions_avg)
            point_query_list = load_query(data_distribution, 0).tolist()
            start_time = time.time()
            index.test_point_query(point_query_list)
            end_time = time.time()
            search_time = (end_time - start_time) / len(point_query_list)
            logging.info("Point query time: %s" % search_time)
            logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
            io_cost = index.io_cost
            range_query_list = load_query(data_distribution, 1).tolist()
            for i in range(len(range_query_list) // 1000):
                tmp_range_query_list = range_query_list[i * 1000:(i + 1) * 1000]
                start_time = time.time()
                index.test_range_query(tmp_range_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / len(tmp_range_query_list)
                logging.info("Range query time: %s" % search_time)
                logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(tmp_range_query_list)))
                io_cost = index.io_cost
            knn_query_list = load_query(data_distribution, 2).tolist()
            for i in range(len(knn_query_list) // 1000):
                tmp_knn_query_list = knn_query_list[i * 1000:(i + 1) * 1000]
                start_time = time.time()
                index.test_knn_query(tmp_knn_query_list)
                end_time = time.time()
                search_time = (end_time - start_time) / len(tmp_knn_query_list)
                logging.info("KNN query time: %s" % search_time)
                logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(tmp_knn_query_list)))
                io_cost = index.io_cost


def grid_search_ts_tm():
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    tn = 10000
    ts_list = [20000, 10000, 5000, 1000]
    tm_list = [100, 50, 10]
    origin_path = "model/slbrin/%s_SORTED_%s"
    target_path = "model/slbrin/%s_SORTED_%s_%s_%s"
    for data_distribution in data_distributions:
        for ts in ts_list:
            for tm in tm_list:
                # 拷贝目标索引磁盘文件
                origin_model_path = origin_path % (data_distribution.name, tn)
                target_model_path = target_path % (data_distribution.name, tn, ts, tm)
                if os.path.exists(target_model_path):
                    shutil.rmtree(target_model_path)
                copy_dirs(origin_model_path, target_model_path, ignore_file='hdf')
                # 加载索引结构到内存
                index = SLBRIN(model_path=target_model_path)
                index.load()
                index.meta.threshold_summary = ts
                index.meta.threshold_merge = tm
                index.meta.threshold_err = 10000
                # 开始更新
                logging.info("*************start %s************" % target_model_path)
                update_data_list = load_data(data_distribution, 1)
                start_time = time.time()
                index.insert(update_data_list)
                index.save()
                end_time = time.time()
                logging.info("Sum up full cr time: %s" % index.sum_up_full_cr_time)
                logging.info("Merge outdated cr time: %s" % index.merge_outdated_cr_time)
                logging.info("Retrain inefficient model time: %s" % index.retrain_inefficient_model_time)
                logging.info("Retrain inefficient model num: %s" % index.retrain_inefficient_model_num)
                update_time = end_time - start_time - \
                              index.sum_up_full_cr_time - index.merge_outdated_cr_time - index.retrain_inefficient_model_time
                logging.info("Update time: %s" % update_time)


def grid_search_te():
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    tn = 10000
    ts = 1000
    tm = 50
    te_list = [10.0, 5.0, 2.0, 1.5, 1.0]
    origin_path = "model/slbrin/%s_SORTED_%s"
    target_path = "model/slbrin/%s_SORTED_%s_%s_%s_%s"
    for te in te_list:
        for data_distribution in data_distributions:
            # 拷贝目标索引磁盘文件
            origin_model_path = origin_path % (data_distribution.name, tn)
            target_model_path = target_path % (data_distribution.name, tn, ts, tm, te)
            if os.path.exists(target_model_path):
                shutil.rmtree(target_model_path)
            target_model_hdf_dir = os.path.join(target_model_path, "hdf/")
            if os.path.exists(target_model_hdf_dir) is False:
                os.makedirs(target_model_hdf_dir)
            copy_dirs(origin_model_path, target_model_path, ignore_file='hdf')
            # 加载索引结构到内存
            index = SLBRIN(model_path=target_model_path)
            index.load()
            index.meta.threshold_summary = ts
            index.meta.threshold_merge = tm
            index.meta.threshold_err = te
            # 开始更新
            logging.info("*************start %s************" % target_model_path)
            update_data_list = load_data(data_distribution, 1)
            start_time = time.time()
            index.insert(update_data_list)
            index.save()
            end_time = time.time()
            logging.info("Sum up full cr time: %s" % index.sum_up_full_cr_time)
            logging.info("Merge outdated cr time: %s" % index.merge_outdated_cr_time)
            logging.info("Retrain inefficient model time: %s" % index.retrain_inefficient_model_time)
            logging.info("Retrain inefficient model num: %s" % index.retrain_inefficient_model_num)
            update_time = end_time - start_time - \
                          index.sum_up_full_cr_time - index.merge_outdated_cr_time - index.retrain_inefficient_model_time
            logging.info("Update time: %s" % update_time)


"""
实验探究：对比不同超参下SLBRIN的整体性能
1. grid_search_w_lr_bs：不同range loss的超参w、学习率lr和批大小bs下，SLBRIN的构建时间
2. grid_search_tn：不同数量阈值tn下，SLBRIN的构建时间、存储成本和检索时间
3. grid_search_ts_tm：不同统计阈值ts、合并阈值tm下，SLBRIN的更新时间
4. grid_search_te：不同误差阈值te下，SLBRIN的更新时间
"""
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parent_path = "model/slbrin"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(filename=os.path.join(parent_path, "log.file"),
                        level=logging.INFO,
                        format="%(message)s")
    # grid_search_w_lr_bs()
    grid_search_tn()
    # grid_search_ts_tm()
    # grid_search_te()
