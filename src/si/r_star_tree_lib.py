import logging
import os
import time

import numpy as np
from rtreelib import RStarTree as RStarTreeLib, Rect

from src.experiment.common_utils import load_data, Distribution
from src.spatial_index import SpatialIndex


class RStarTree(SpatialIndex):
    """
    R星树（R*-tree）
    Implement from pypi: rtreelib
    """

    def __init__(self, model_path=None):
        super(RStarTree, self).__init__("RStarTree")
        self.model_path = model_path
        self.index = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def insert_single(self, point):
        self.index.insert(point[-1], Rect(point[0], point[1], point[0], point[1]))

    def delete(self, point):
        return

    def build(self, data_list, threshold_number):
        self.index = RStarTreeLib(max_entries=threshold_number)
        self.insert(data_list)

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


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/r_star_tree_lib_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = RStarTree(model_path=model_path)
    index_name = index.name
    load_index_from_file = False
    if load_index_from_file:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(Distribution.NYCT_10W, 0)
        index.build(data_list=build_data_list, threshold_number=100)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    path = '../../data/query/point_query_nyct.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/range_query_nyct.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/knn_query_nyct.npy'
    knn_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    # update_data_list = load_data(Distribution.NYCT_10W, 1)
    # start_time = time.time()
    # index.insert(update_data_list)
    # end_time = time.time()
    # logging.info("Update time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
