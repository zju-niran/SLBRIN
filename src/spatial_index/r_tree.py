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

    def search(self, point):
        search_result = self.index.intersection((point.lng, point.lat))
        return list(search_result)

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
            results.append(self.search(Point(point[0], point[1]))[0])
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
    train_set_xy_list = np.delete(train_set_xy.values, 0, 1).tolist()
    start_time = time.time()
    result = index.point_query(train_set_xy_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(train_set_xy_list)
    print("Search time ", search_time)
    print("Not found nums ", pd.Series(result).isna().sum())
    print("*************end %s************" % index_name)


if __name__ == '__main__':
    main()
