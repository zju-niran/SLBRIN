import os
import random

from src.common_utils import Point, read_data_and_search, Region
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
                if node.items[i] == point:
                    delete_index = node.items[i].index
                    combine_flag = True
                    del node.items[i]
                    return delete_index, combine_flag
            return -1, combine_flag
        else:
            y_center = (node.region.up + node.region.bottom) / 2
            x_center = (node.region.left + node.region.right) / 2
            if point.lat > y_center:
                if point.lng > x_center:
                    delete_index, combine_flag = self.delete(point, node.RU)
                else:
                    delete_index, combine_flag = self.delete(point, node.LU)
            else:
                if point.lng > x_center:
                    delete_index, combine_flag = self.delete(point, node.RB)
                else:
                    delete_index, combine_flag = self.delete(point, node.LB)
            if combine_flag:
                if (len(node.RU.items) + len(node.LU.items) + len(node.RB.items) + len(node.LB.items)) <= self.max_num:
                    self.combine_node(node)
                    combine_flag = False
            return delete_index, combine_flag

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
            for item in node.items:
                if item == point:
                    # if point.near(item):
                    return item.index
            return -1

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



if __name__ == '__main__':
    os.chdir('D:\\Code\\Paper\\st-learned-index')
    path = 'data/test_x_y_index.csv'
    index = QuadTree()
    read_data_and_search(path, index)
