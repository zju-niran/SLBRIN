import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from rtreelib import RStarTree as RStarTreeLib, Rect

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.spatial_index import SpatialIndex
from src.spatial_index.common_utils import Point


class RStarTree(SpatialIndex):
    def __init__(self, model_path=None):
        super(RStarTree, self).__init__("RStarTree")
        self.model_path = model_path
        self.index = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def insert(self, point):
        self.index.insert(point.index, Rect(point.lng, point.lat, point.lng, point.lat))

    def delete(self, point):
        return

    def build(self, data_list, threshold_number):
        self.index = RStarTreeLib(max_entries=threshold_number)
        for i in range(len(data_list)):
            self.insert(Point(data_list[i][0], data_list[i][1], index=i))

    def point_query_single(self, point):
        """
        query index by x/y point
        1. search by x/y
        2. for duplicate point: only return the first one
        """
        return [i.data for i in self.index.query((point[0], point[1]))]

    def range_query_single(self, window):
        """
        query index by x1/y1/x2/y2 window
        """
        return [i.data for i in self.index.query((window[2], window[0], window[3], window[1]))]

    def knn_query_single(self, knn):
        """
        query index by x1/y1/n knn
        """
        return None

    def save(self):
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    # def load(self):


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/trip_data_1_10w.npy'
    model_path = "model/r_star_tree_lib_10w/"
    index = RStarTree(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        data_list = np.load(data_path).tolist()
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        index.build(data_list=data_list, threshold_number=100)
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
        index.save()
    path = '../../data/trip_data_1_point_query.csv'
    point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
    point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time:  %s" % search_time)
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


if __name__ == '__main__':
    main()
