import heapq
import os
import sys
import time

import pandas as pd
from memory_profiler import profile

sys.path.append('D:/Code/Paper/st-learned-index')
from src.index import Index
from src.spatial_index.common_utils import Region, Point

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
        self.LB = None
        self.RB = None
        self.LU = None
        self.RU = None
        self.items = []


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
        if point.lat < y_center:
            if point.lng < x_center:
                self.insert(point, node.LB)
            else:
                self.insert(point, node.RB)
        else:
            if point.lng < x_center:
                self.insert(point, node.LU)
            else:
                self.insert(point, node.RU)

    def split_node(self, node):
        """
        分裂节点
        1.通过父节点获取子节点的深度和范围
        2.生成四个节点，挂载到父节点下
        """
        y_center = (node.region.up + node.region.bottom) / 2
        x_center = (node.region.left + node.region.right) / 2

        node.is_leaf = 0
        node.LB = self.create_child_node(node, node.region.bottom, y_center, node.region.left, x_center)
        node.RB = self.create_child_node(node, node.region.bottom, y_center, x_center, node.region.right)
        node.LU = self.create_child_node(node, y_center, node.region.up, node.region.left, x_center)
        node.RU = self.create_child_node(node, y_center, node.region.up, x_center, node.region.right)

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
        node.items = node.LB.items + node.RB.items + node.LU.items + node.RU.items
        node.LB = None
        node.RB = None
        node.LU = None
        node.RU = None

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
        if point.lat < y_center:
            if point.lng < x_center:
                return self.search(point, node.LB)
            else:
                return self.search(point, node.RB)
        else:
            if point.lng < x_center:
                return self.search(point, node.LU)
            else:
                return self.search(point, node.RU)

    def search_node(self, point, node=None):
        """
        找到point所在的node
        :param point:
        :param node:
        :return:
        """
        if node is None:
            node = self.root_node
        if node.is_leaf == 1:
            return node
        y_center = round((node.region.up + node.region.bottom) / 2, self.data_precision)
        x_center = round((node.region.left + node.region.right) / 2, self.data_precision)
        if point.lat < y_center:
            if point.lng < x_center:
                return self.search_node(point, node.LB)
            else:
                return self.search_node(point, node.RB)
        else:
            if point.lng < x_center:
                return self.search_node(point, node.LU)
            else:
                return self.search_node(point, node.RU)

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
            self.geohash(node.RB, parent_geohash + "01")
            self.geohash(node.LU, parent_geohash + "10")
            self.geohash(node.RU, parent_geohash + "11")

    def build(self, data: pd.DataFrame, z=False):
        if z is False:
            for index, point in data.iterrows():
                self.insert(Point(point.x, point.y, index=index))
        else:
            for index, point in data.iterrows():
                self.insert(Point(point.x, point.y, point.z, index))

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

    def range_search(self, region, node=None, result: list = []):
        if node is None:
            node = self.root_node
        if node.is_leaf == 1:
            for item in node.items:
                if region.contain_and_border(item):
                    result.append(item.index)
        else:
            # 所有的or：region的四至点刚好在子节点的region上，因为split的时候经纬度都是向上取整，所以子节点的重心在右和上
            if node.LB.region.contain(Point(region.left, region.bottom)):
                self.range_search(Region(region.bottom, min(node.LB.region.up, region.up),
                                         region.left, min(node.LB.region.right, region.right)),
                                  node.LB, result)
            if node.RB.region.contain(Point(region.right, region.bottom)) \
                    or (region.bottom < node.RB.region.up and region.right == node.RB.region.right):
                self.range_search(Region(region.bottom, min(node.LB.region.up, region.up),
                                         max(node.RU.region.left, region.left), region.right),
                                  node.RB, result)
            if node.LU.region.contain(Point(region.left, region.up)) \
                    or (region.left < node.LU.region.right and region.up == node.LU.region.up):
                self.range_search(Region(max(node.RU.region.bottom, region.bottom), region.up,
                                         region.left, min(node.LB.region.right, region.right)),
                                  node.LU, result)
            if node.RU.region.contain(Point(region.right, region.up)) \
                    or (region.right > node.RU.region.left and region.up == node.RU.region.up) \
                    or (region.up > node.RU.region.bottom and region.right == node.RU.region.right):
                self.range_search(Region(max(node.RU.region.bottom, region.bottom), region.up,
                                         max(node.RU.region.left, region.left), region.right),
                                  node.RU, result)

    def range_query(self, windows):
        """
        query index by x1/y1/x2/y2 window
        :param windows: list, [x1, y1, x2, y2]
        :return: list, [pres]
        """
        results = []
        for window in windows:
            result = []
            region = Region(window[0], window[1], window[2], window[3])
            self.range_search(region=region, node=None, result=result)
            results.append(result)
        return results

    def knn_query_old(self, knns):
        """
        query index by x1/y1/n knn
        代码参考：https://github.com/diana12333/QuadtreeNN
        1.用root node初始化stack，nearest_distance和point_heap分别为正无穷和空
        2.循环：当stack非空
        2.1.如果节点距离够：node.region和point的距离不超过nearest_distance
        2.1.1.如果node is_leaf，则遍历items，用距离够的item更新point_heap，同时更新nearest_distance
        2.1.2.如果node not_leaf，则把child放入stack
        3.返回result里的所有index
        :param knns: list, [x1, y1, n]
        :return: list, [pres]
        """
        results = []
        for knn in knns:
            point = Point(knn[0], knn[1])
            n = knn[2]
            stack = [self.root_node]
            nearest_distance = (float('-inf'), None)
            point_heap = []
            while len(stack):
                cur = stack.pop(-1)
                if cur.region.within_distance_pow(point, -nearest_distance[0]):
                    if cur.is_leaf:
                        for item in cur.items:
                            if len(point_heap) < n:
                                heapq.heappush(point_heap, (-point.distance_pow(item), item.index))
                                nearest_distance = heapq.nsmallest(1, point_heap)[0]
                                continue
                            point_distance = point.distance_pow(item)
                            if point_distance < -nearest_distance[0]:
                                heapq.heappop(point_heap)
                                heapq.heappush(point_heap, (-point_distance, item.index))
                                nearest_distance = heapq.nsmallest(1, point_heap)[0]
                    else:
                        stack.append(cur.LB)
                        stack.append(cur.RB)
                        stack.append(cur.LU)
                        stack.append(cur.RU)
            results.append([itr[1] for itr in point_heap])
        return results

@profile(precision=8)
    def knn_query(self, knns):
        """
        query index by x1/y1/n knn
        1.先找到point所在的节点，初始化nearest_distance和point_heap
        2.后续操作和knn_query_old一致，但是由于nearest_distance被初始化，后续遍历可以减少大量节点的距离判断
        检索时间从0.099225优化到0.006712
        :param knns: list, [x1, y1, n]
        :return: list, [pres]
        """
        results = []
        for knn in knns:
            point = Point(knn[0], knn[1])
            n = knn[2]
            stack = [self.root_node]
            nearest_distance = (float('-inf'), None)
            point_heap = []
            point_node = self.search_node(point)
            for item in point_node.items:
                if len(point_heap) < n:
                    heapq.heappush(point_heap, (-point.distance_pow(item), item.index))
                    nearest_distance = heapq.nsmallest(1, point_heap)[0]
                    continue
                point_distance = point.distance_pow(item)
                if point_distance < -nearest_distance[0]:
                    heapq.heappop(point_heap)
                    heapq.heappush(point_heap, (-point_distance, item.index))
                    nearest_distance = heapq.nsmallest(1, point_heap)[0]
            while len(stack):
                cur = stack.pop(-1)
                # 跳过point所在节点的判断
                if cur == point_node:
                    continue
                if cur.region.within_distance_pow(point, -nearest_distance[0]):
                    if cur.is_leaf:
                        for item in cur.items:
                            if len(point_heap) < n:
                                heapq.heappush(point_heap, (-point.distance_pow(item), item.index))
                                nearest_distance = heapq.nsmallest(1, point_heap)[0]
                                continue
                            point_distance = point.distance_pow(item)
                            if point_distance < -nearest_distance[0]:
                                heapq.heappop(point_heap)
                                heapq.heappush(point_heap, (-point_distance, item.index))
                                nearest_distance = heapq.nsmallest(1, point_heap)[0]
                    elif not cur.is_leaf:
                        stack.append(cur.LB)
                        stack.append(cur.RB)
                        stack.append(cur.LU)
                        stack.append(cur.RU)
            results.append([itr[1] for itr in point_heap])
        return results


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # load data
    path = '../../data/trip_data_1_filter.csv'
    train_set_xy = pd.read_csv(path)
    # create index
    model_path = "model/quadtree_1451w/"
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
    print("*************start point query************")
    point_query_list = train_set_xy.drop("index", axis=1).values.tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    print("Point query time ", search_time)
    print("Not found nums ", pd.Series(results).isna().sum())
    print("*************start range query************")
    path = '../../data/trip_data_1_range_query.csv'
    range_query_df = pd.read_csv(path, usecols=[1, 2, 3, 4, 5])
    range_query_list = range_query_df.drop("count", axis=1).values.tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    print("Range query time ", search_time)
    range_query_df["query"] = pd.Series(results).apply(len)
    print("Not found nums ", (range_query_df["query"] != range_query_df["count"]).sum())
    print("*************start knn query************")
    path = '../../data/trip_data_1_knn_query.csv'
    knn_query_df = pd.read_csv(path, usecols=[1, 2, 3], dtype={"n": int})
    knn_query_list = [[value[0], value[1], int(value[2])] for value in knn_query_df.values]
    start_time = time.time()
    results1 = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    print("KNN query time ", search_time)
    print("*************end %s************" % index_name)


if __name__ == '__main__':
    main()
