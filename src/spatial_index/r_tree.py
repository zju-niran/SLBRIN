import os
import time

import pandas as pd
from rtree import index

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

    def point_query(self, data: pd.DataFrame):
        """
        query index by x/y point
        1. search by x/y
        2. for duplicate point: only return the first one
        :param data: pd.DataFrame, [x, y]
        :return: pd.DataFrame, [pre]
        """
        results = data.apply(lambda t: self.search(Point(t.x, t.y))[0], 1)
        return results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_2_100000_random.csv'
    # read_data_and_search(path, index, None, None, 7, 8)
    z_col, index_col = 7, 8
    train_set_xy = pd.read_csv(path, header=None, usecols=[2, 3], names=["x", "y"])
    test_ratio = 0.5  # 测试集占总数据集的比例
    test_set_xy = train_set_xy.sample(n=int(len(train_set_xy) * test_ratio), random_state=1)
    # create index
    model_path = "model/zm_index_2022-01-25/"
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
    start_time = time.time()
    result = index.point_query(test_set_xy)
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_xy)
    print("Search time ", search_time)
    print("Not found nums ", result.isna().sum())
    print("*************end %s************" % index_name)
