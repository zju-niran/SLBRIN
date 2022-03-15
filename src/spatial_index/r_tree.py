import os
import sys
import time

import numpy as np
import pandas as pd
from rtree import index

sys.path.append('D:/Code/Paper/st-learned-index')
from src.index import Index
from src.spatial_index.common_utils import Point


class RTree(Index):
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

    def point_query(self, points):
        """
        query index by x/y point
        1. search by x/y
        2. for duplicate point: only return the first one
        :param points: list, [x, y]
        :return: list, [pre]
        """
        return [list(self.index.intersection((point[0], point[1]))) for point in points]

    def range_query(self, windows):
        """
        query index by x1/y1/x2/y2 window
        :param windows: list, [x1, y1, x2, y2]
        :return: list, [pres]
        """
        return [list(self.index.intersection((window[2], window[0], window[3], window[1]))) for window in windows]

    def knn_query(self, knns):
        """
        query index by x1/y1/n knn
        :param knns: list, [x1, y1, n]
        :return: list, [pres]
        """
        return [list(self.index.nearest((knn[0], knn[1]), knn[2])) for knn in knns]

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
    path = '../../data/trip_data_1_filter.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/rtree_1451w/"
    index = RTree(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(train_set_xy)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        index.save()
    path = '../../data/trip_data_1_point_query.csv'
    point_query_df = pd.read_csv(path, usecols=[1, 2, 3])
    point_query_list = point_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    print("Point query time ", search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    print("Range query time ", search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/trip_data_1_knn_query.csv'
    knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    print("KNN query time ", search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    print("*************end %s************" % index_name)


if __name__ == '__main__':
    main()
