import os

from rtree import index

from src.common_utils import read_data_and_search
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
        return list(search_result)

    def delete(self, point):
        self.index.delete(point.index, (point.lng, point.lat))


if __name__ == '__main__':
    os.chdir('D:\\Code\\Paper\\st-learned-index')
    path = 'data/test_x_y_index.csv'
    index = RTree()
    read_data_and_search(path, index)
