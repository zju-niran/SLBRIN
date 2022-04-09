import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from rtree import index

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.spatial_index import SpatialIndex
from src.spatial_index.common_utils import Point


class RTree(SpatialIndex):
    def __init__(self, model_path=None):
        super(RTree, self).__init__("RTree")
        self.model_path = model_path
        self.index = None
        self.threshold_number = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def insert(self, point):
        self.index.insert(point.key, (point.lng, point.lat))

    def delete(self, point):
        self.index.delete(point.key, (point.lng, point.lat))

    def build(self, data_list, threshold_number):
        self.threshold_number = threshold_number
        p = index.Property()
        p.dimension = 2
        p.dat_extension = "data"
        p.idx_extension = "key"
        p.storage = index.RT_Disk
        p.pagesize = threshold_number
        p.leaf_capacity = threshold_number
        self.index = index.Index(os.path.join(self.model_path, 'rtree'), properties=p, overwrite=True)
        # self.index = index.RtreeContainer(properties=p)  # 没有直接Index来得快，range_query慢了一倍
        for i in range(len(data_list)):
            self.insert(Point(data_list[i][0], data_list[i][1], key=i))

    def point_query_single(self, point):
        """
        query key by x/y point
        1. search by x/y
        2. for duplicate point: only return the first one
        """
        return list(self.index.intersection((point[0], point[1])))

    def range_query_single(self, window):
        """
        query key by x1/y1/x2/y2 window
        """
        return list(self.index.intersection((window[2], window[0], window[3], window[1])))

    def knn_query_single(self, knn):
        """
        query key by x1/y1/n knn
        """
        return list(self.index.nearest((knn[0], knn[1]), knn[2]))

    def save(self):
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        with open(self.model_path + 'rtree.json', "w") as f:
            json.dump(self.threshold_number, f, ensure_ascii=False)

    def load(self):
        with open(self.model_path + 'rtree.json', "r") as f:
            threshold_number = json.load(f)
        p = index.Property()
        p.dimension = 2
        p.dat_extension = "data"
        p.idx_extension = "key"
        p.storage = index.RT_Disk
        p.pagesize = threshold_number
        p.leaf_capacity = threshold_number
        self.index = index.Index(os.path.join(self.model_path, 'rtree'),
                                 interleaved=False, properties=p, overwrite=False)

    def size(self):
        """
        size = rtree.data + rtree.key + rtree.json
        """
        return os.path.getsize(os.path.join(self.model_path, "rtree.data")) + \
               os.path.getsize(os.path.join(self.model_path, "rtree.key")) + \
               os.path.getsize(os.path.join(self.model_path, "rtree.json"))


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/trip_data_1_filter_sorted.npy'
    model_path = "model/rtree_1451w/"
    index = RTree(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        data_list = np.load(data_path).tolist()
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        index.build(data_list=data_list, threshold_number=100)
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time %s" % build_time)
        index.save()
    logging.info("Index size: %s" % index.size())
    path = '../../data/trip_data_1_point_query.csv'
    point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
    point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    index.test_point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    index.logging.info("Point query time %s" % search_time)
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    index.test_range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    index.logging.info("Range query time %s" % search_time)
    path = '../../data/trip_data_1_knn_query.csv'
    knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    start_time = time.time()
    index.test_knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    index.logging.info("KNN query time %s" % search_time)
    index.logging.info("*************end %s************" % index_name)


if __name__ == '__main__':
    main()
