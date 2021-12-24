import os
import random

from rtree import index

from src.common_utils import Point
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
        return self.index.delete(point.index, (point.lng, point.lat))


def create_data_and_search():
    r_tree = RTree()
    lat, lng, index = 0, 0, 0
    for i in range(101):
        lng = random.uniform(-180, 180)
        lat = random.uniform(-90, 90)
        index += 1
        r_tree.insert(Point(lng, lat, index))
    test_point = Point(lng, lat)
    search_index = r_tree.search(test_point)
    print("{0} is found in {1}".format(test_point, search_index))
    delete_index = r_tree.delete(test_point)
    print("{0} is deleted in {1}".format(test_point, delete_index))


if __name__ == '__main__':
    os.chdir('D:\\Code\\Paper\\st-learned-index')
    path = 'data/test_x_y_index.csv'
    # create_data(path)
    create_data_and_search()
    # read_data_and_search(path)
