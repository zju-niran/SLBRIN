import os
import sys
import time

import pandas as pd
from memory_profiler import profile

sys.path.append('D:/Code/Paper/st-learned-index')
from src.index import Index
from src.spatial_index.common_utils import Region, Point, Geohash

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
        :param region: 四叉树整体的bbox
        :param max_num: 节点内的点数据数量预置
        """
        super(QuadTree, self).__init__("QuadTree")
        self.max_num = max_num
        self.root_node = QuadTreeNode(region=region)
        self.geohash_items_map = {}

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

    def geohash(self, node=None, parent_geohash=None):
        """
        get geohash->items by quad tree
        :param node: for iter
        :param parent_geohash: for iter
        :return: save geohash->items in self.geohash_data_map
        """
        if node is None:
            node = self.root_node
        if parent_geohash is None:
            parent_geohash = ""
        # 节点内部查找：遍历
        if node.is_leaf == 1:
            if len(node.items) > 0:
                sorted_items = sorted(node.items, key=lambda point: point.z)
                self.geohash_items_map[parent_geohash] = {
                    "z_border": [sorted_items[0].z, sorted_items[-1].z],
                    "xy_border": node.region,
                    "items": node.items
                }
            return
        else:
            self.geohash(node.LB, parent_geohash + "00")
            self.geohash(node.LU, parent_geohash + "01")
            self.geohash(node.RB, parent_geohash + "10")
            self.geohash(node.RU, parent_geohash + "11")

    def build(self, data: pd.DataFrame, z=False):
        if z is False:
            for index, point in data.iterrows():
                self.insert(Point(point.x, point.y, index=index))
        else:
            for index, point in data.iterrows():
                self.insert(Point(point.x, point.y, point.z, point.z_index))

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



@profile(precision=8)
def main():
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
    index = QuadTree(region=Region(40, 42, -75, -73), max_num=1000)
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


if __name__ == '__main__':
    main()
