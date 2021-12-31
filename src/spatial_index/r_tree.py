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

    def predict(self, point):
        """
        预测point并且计算误差
        1. 如果预测的list包含正确的index，则误差为0
        2. 如果预测的list不包含正确的index，则误差为所有预测位置到index的距离和
        :param point: 预测点
        :return: 误差
        """
        pre_list = self.search(point)
        err = 0
        for pre in pre_list:
            if pre == point.index:
                return 0
            else:
                err += abs(pre - point.index)
        return err

    def delete(self, point):
        self.index.delete(point.index, (point.lng, point.lat))


if __name__ == '__main__':
    os.chdir('/')
    path = 'data/test_x_y_index.csv'
    index = RTree()
    read_data_and_search(path, index)
