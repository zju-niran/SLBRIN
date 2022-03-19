import logging
import os
import sys
import time

import pandas as pd
from rtree import index

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.spatial_index import SpatialIndex
from src.spatial_index.common_utils import Point


class RTree(SpatialIndex):
    def __init__(self, model_path=None):
        super(RTree, self).__init__("RTree")
        self.model_path = model_path
        p = index.Property()
        self.index = index.Index(properties=p)

    def insert(self, point):
        self.index.insert(point.index, (point.lng, point.lat))

    def delete(self, point):
        self.index.delete(point.index, (point.lng, point.lat))

    def build(self, data: pd.DataFrame):
        for index, point in data.iterrows():
            self.insert(Point(point.x, point.y, index=index))

    def point_query_single(self, point):
        """
        query index by x/y point
        1. search by x/y
        2. for duplicate point: only return the first one
        """
        return list(self.index.intersection((point[0], point[1])))

    def range_query_single(self, window):
        """
        query index by x1/y1/x2/y2 window
        """
        return list(self.index.intersection((window[2], window[0], window[3], window[1])))

    def knn_query_single(self, knn):
        """
        query index by x1/y1/n knn
        """
        return list(self.index.nearest((knn[0], knn[1]), knn[2]))

    def save(self):
        """
        save rtree into json file
        :return: None
        """
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def load(self):
        """
        load zm index from json file
        :return: None
        """


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_1_100000.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/rtree_10w/"
    index = RTree(model_path=model_path)
    index_name = index.name
    logging.basicConfig(filename=os.path.join(model_path, "log.file"),
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S %p")
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        logging.info("*************start %s************" % index_name)
        start_time = time.time()
        index.build(train_set_xy)
        end_time = time.time()
        build_time = end_time - start_time
        logging.info("Build time %s" % build_time)
        index.save()
    path = '../../data/trip_data_1_point_query.csv'
    point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
    point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    index.test_point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time %s" % search_time)
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    index.test_range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time %s" % search_time)
    path = '../../data/trip_data_1_knn_query.csv'
    knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    start_time = time.time()
    index.test_knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time %s" % search_time)
    logging.info("*************end %s************" % index_name)


if __name__ == '__main__':
    main()
