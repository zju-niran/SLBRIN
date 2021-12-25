import os
import random

from rtree import index

from src.common_utils import Point, read_data_and_search
from src.index import Index


class RTree(Index):
    def __init__(self):
        super(RTree, self).__init__("RTree")
        p = index.Property()
        self.index = index.Index(properties=p)

    def insert(self, point):
        self.index.insert(point.index, (point.lng, point.lat))

    def search(self, point):
        search_result = self.index.intersection((point.lng, point.lat))
        search_result_list = list(search_result)
        return search_result_list[0] if len(search_result_list) != 0 else -1

    def delete(self, point):
        self.index.delete(point.index, (point.lng, point.lat))


if __name__ == '__main__':
    os.chdir('D:\\Code\\Paper\\st-learned-index')
    path = 'data/test_x_y_index.csv'
    index = RTree()
    read_data_and_search(path, index)
