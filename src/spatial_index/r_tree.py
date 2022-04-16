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
        self.leaf_node_capacity = None
        self.internal_node_capacity = None
        self.buffering_capacity = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def insert(self, point):
        self.index.insert(point.key, (point.lng, point.lat))

    def insert_batch(self, points):
        for i in range(len(points)):
            self.insert(Point(points[i][0], points[i][1], key=i))

    def delete(self, point):
        self.index.delete(point.key, (point.lng, point.lat))

    def build(self, data_list, leaf_node_capacity, internal_node_capacity, buffering_capacity):
        self.leaf_node_capacity = leaf_node_capacity
        self.internal_node_capacity = internal_node_capacity
        self.buffering_capacity = buffering_capacity
        p = index.Property()
        p.dimension = 2
        p.dat_extension = "data"
        p.idx_extension = "key"
        p.storage = index.RT_Disk
        # TODO: 增大buffer可以提高查询效率，而且10w数据下build和insert影响不大，当前单位Byte
        p.buffering_capacity = buffering_capacity
        # sys.getsizeof(0) * 4表示一个MBR的内存大小，也可以通过sys.getsizeof(index.bounds)获取
        p.pagesize = internal_node_capacity * sys.getsizeof(0) * 4
        p.leaf_capacity = leaf_node_capacity
        self.index = index.Index(os.path.join(self.model_path, 'rtree'), properties=p, overwrite=True)
        # self.index = index.RtreeContainer(properties=p)  # 没有直接Index来得快，range_query慢了一倍
        self.insert_batch(data_list)

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
        return list(self.index.nearest((knn[0], knn[1]), knn[2]))[:knn[2]]

    def save(self):
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        with open(self.model_path + 'rtree.json', "w") as f:
            rtree_meta_json = {
                "leaf_node_capacity": self.leaf_node_capacity,
                "internal_node_capacity": self.internal_node_capacity,
                "buffering_capacity": self.buffering_capacity
            }
            json.dump(rtree_meta_json, f, ensure_ascii=False)

    def load(self):
        with open(self.model_path + 'rtree.json', "r") as f:
            rtree_meta_json = json.load(f)
        p = index.Property()
        p.dimension = 2
        p.dat_extension = "data"
        p.idx_extension = "key"
        p.storage = index.RT_Disk
        p.pagesize = rtree_meta_json["internal_node_capacity"] * sys.getsizeof(0) * 4
        p.leaf_capacity = rtree_meta_json["leaf_node_capacity"]
        p.buffering_capacity = rtree_meta_json["buffering_capacity"]
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
    data_path = '../../data/trip_data_1_10w.npy'
    model_path = "model/rtree_10w/"
    index = RTree(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        data_list = np.load(data_path).tolist()
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        index.build(data_list=data_list,
                    leaf_node_capacity=100,
                    internal_node_capacity=100,
                    buffering_capacity=1024)
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
        index.save()
    logging.info("Index size: %s" % index.size())
    path = '../../data/trip_data_1_point_query.csv'
    point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
    point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time:  %s" % search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/trip_data_1_knn_query.csv'
    knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time:  %s" % search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    insert_data_list = np.load("../../data/trip_data_2_10w.npy").tolist()
    start_time = time.time()
    index.insert_batch(insert_data_list)
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
