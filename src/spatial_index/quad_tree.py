import os
import time

import pandas as pd

from src.spatial_index.common_utils import Region, Point, Geohash
# settings
from src.index import Index

MAX_ELE_NUM = 100

QUADRANT_RU = 1
QUADRANT_LU = 2
QUADRANT_LB = 3
QUADRANT_RB = 4


class QuadTreeNode:
    def __init__(self, region, depth=1, is_leaf=1):
        self.depth = depth
        self.is_leaf = is_leaf
        self.region = region
        self.LU = None
        self.LB = None
        self.RU = None
        self.RB = None
        self.items = []  # ElePoitems[MAX_ELE_NUM]


class QuadTree(Index):
    def __init__(self, region=Region(-90, 90, -180, 180), max_num=MAX_ELE_NUM):
        """
        初始化非满四叉树，超过阈值就分裂
        :param max_num: 节点内的点数据数量预置
        """
        super(QuadTree, self).__init__("QuadTree")
        self.max_num = max_num
        self.root_node = QuadTreeNode(region=region)
        self.leaf_list = []
        self.node_geohash = None

    def insert(self, point, node=None):
        """
        插入元素
        1.判断是否已分裂，已分裂的选择适合的子节点，插入；
        2.未分裂的查看是否过载，过载的分裂节点，重新插入；
        3.未过载的直接添加
    
        @param node
        @param point
    
        todo 使用元素原地址，避免重新分配内存造成的效率浪费
        """
        if node is None:
            node = self.root_node
        if node.is_leaf == 1:
            if len(node.items) + 1 > self.max_num:
                self.split_node(node)
                self.insert(point, node)
            else:
                # todo 点排重（不排重的话如果相同的点数目大于 MAX_ELE_NUM， 会造成无限循环分裂）
                node.items.append(point)
            return

        y_center = (node.region.up + node.region.bottom) / 2
        x_center = (node.region.left + node.region.right) / 2
        if point.lat > y_center:
            if point.lng > x_center:
                self.insert(point, node.RU)
            else:
                self.insert(point, node.LU)
        else:
            if point.lng > x_center:
                self.insert(point, node.RB)
            else:
                self.insert(point, node.LB)

    def split_node(self, node):
        """
        分裂节点
        1.通过父节点获取子节点的深度和范围
        2.生成四个节点，挂载到父节点下
        """
        y_center = (node.region.up + node.region.bottom) / 2
        x_center = (node.region.left + node.region.right) / 2

        node.is_leaf = 0
        node.RU = self.create_child_node(node, y_center, node.region.up, x_center, node.region.right)
        node.LU = self.create_child_node(node, y_center, node.region.up, node.region.left, x_center)
        node.RB = self.create_child_node(node, node.region.bottom, y_center, x_center, node.region.right)
        node.LB = self.create_child_node(node, node.region.bottom, y_center, node.region.left, x_center)

        for item in node.items:
            self.insert(item, node)

        # 清空父节点的element
        node.items = None

    def create_child_node(self, node, bottom, up, left, right):
        depth = node.depth + 1
        region = Region(bottom, up, left, right)
        child_node = QuadTreeNode(region=region, depth=depth)
        return child_node

    def delete(self, point, node=None):
        """
        删除元素
        1. 遍历元素列表，删除对应元素
        2. 检查兄弟象限元素总数，不超过最大量时组合兄弟象限
        """
        combine_flag = False
        if node is None:
            node = self.root_node
        if node.is_leaf == 1:
            for i in range(len(node.items)):
                if node.items[i] == point and node.items[i].index == point.index:
                    combine_flag = True
                    del node.items[i]
            return combine_flag
        else:
            y_center = (node.region.up + node.region.bottom) / 2
            x_center = (node.region.left + node.region.right) / 2
            if point.lat > y_center:
                if point.lng > x_center:
                    combine_flag = self.delete(point, node.RU)
                else:
                    combine_flag = self.delete(point, node.LU)
            else:
                if point.lng > x_center:
                    combine_flag = self.delete(point, node.RB)
                else:
                    combine_flag = self.delete(point, node.LB)
            if combine_flag:
                if (len(node.RU.items) + len(node.LU.items) + len(node.RB.items) + len(node.LB.items)) <= self.max_num:
                    self.combine_node(node)
                    combine_flag = False
            return combine_flag

    def combine_node(self, node):
        """
        合并节点
        1. 遍历四个子象限的点，添加到象限点列表
        2. 释放子象限的内存
        """
        node.is_leaf = 1
        node.items = node.RU.items + node.LU.items + node.RB.items + node.LB.items
        node.RU = None
        node.LU = None
        node.RB = None
        node.LB = None

    def search(self, point, node=None):
        if node is None:
            node = self.root_node
        # 节点内部查找：遍历
        if node.is_leaf == 1:
            search_result = []
            for item in node.items:
                if item == point:
                    # if point.near(item):
                    search_result.append(item.index)
            return search_result

        y_center = (node.region.up + node.region.bottom) / 2
        x_center = (node.region.left + node.region.right) / 2
        if point.lat > y_center:
            if point.lng > x_center:
                return self.search(point, node.RU)
            else:
                return self.search(point, node.LU)
        else:
            if point.lng > x_center:
                return self.search(point, node.RB)
            else:
                return self.search(point, node.LB)

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

    def geohash(self, node=None, parent_geohash=None):
        if node is None:
            node = self.root_node
        if parent_geohash is None:
            parent_geohash = ""
        # 节点内部查找：遍历
        if node.is_leaf == 1:
            self.leaf_list.append(parent_geohash)
            node.geohash = parent_geohash
            return
        else:
            self.geohash(node.LB, parent_geohash + "00")
            self.geohash(node.LU, parent_geohash + "01")
            self.geohash(node.RB, parent_geohash + "10")
            self.geohash(node.RU, parent_geohash + "11")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    path = '../../data/test_x_y_index.csv'
    index = QuadTree()
    # read_data_and_search(path, index, 1, 2, None, 0)
    data = pd.read_csv(path, header=None)
    train_set_point = []
    for i in range(int(data.shape[0])):
        geohash = Geohash()
        geohash_value = geohash.encode(data.iloc[i, 1], data.iloc[i, 2], 24)
        train_set_point.append(Point(data.iloc[i, 1], data.iloc[i, 2], geohash_value, data.iloc[i, 0]))
    index.build(train_set_point)
    index.geohash()
    geohash_list = index.leaf_list
    geohash_max_length = max([len(geohash_list_member) for geohash_list_member in geohash_list])
    test_ratio = 0.5  # 测试集占总数据集的比例
    test_set_point = train_set_point[:int(len(train_set_point) * test_ratio)]
    err = 0
    start_time = time.time()
    for ind in range(len(test_set_point)):
        err += index.predict(test_set_point[ind])
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_point)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_point)
    print("mean error = ", mean_error)
