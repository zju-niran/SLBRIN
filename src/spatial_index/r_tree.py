import os
import sys
import time

import numpy as np
import pandas as pd
from memory_profiler import profile
from rtree import index

sys.path.append('D:/Code/Paper/st-learned-index')
from src.index import Index
from src.spatial_index.common_utils import Point


class RTree(Index):
    def __init__(self):
        super(RTree, self).__init__("RTree")
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
        results = []
        for point in points:
            results.append(list(self.index.intersection((point[0], point[1])))[0])
        return results

    def range_query(self, windows):
        """
        query index by x1/y1/x2/y2 window
        :param windows: list, [x1, y1, x2, y2]
        :return: list, [pres]
        """
        results = []
        for window in windows:
            results.append(list(self.index.intersection((window[2], window[0], window[3], window[1]))))
        return results


@profile(precision=8)
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_1_filter.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/rtree_1451w/"
    index = RTree()
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()  # TODO: create load
    else:
        print("*************start %s************" % index_name)
        print("Start Build")
        start_time = time.time()
        index.build(train_set_xy)
        end_time = time.time()
        build_time = end_time - start_time
        print("Build %s time " % index_name, build_time)
        # index.save()  # TODO: create save
    print("*************start point query************")
    point_query_list = np.delete(train_set_xy.values, 0, 1).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    print("Point query time ", search_time)
    print("Not found nums ", pd.Series(results).isna().sum())
    print("*************start range query************")
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path)
    range_query_list = np.delete(range_query_df.values, [0, -1], 1).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    print("Range query time ", search_time)
    range_query_df["query"] = pd.Series(results).apply(len)
    print("Not found nums ", (range_query_df["query"] != range_query_df["count"]).sum())
    print("*************end %s************" % index_name)


if __name__ == '__main__':
    main()
