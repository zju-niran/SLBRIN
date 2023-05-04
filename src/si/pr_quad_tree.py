import copy
import heapq
import logging
import math
import os
import time

import numpy as np

from src.experiment.common_utils import Distribution, load_data, data_precision, data_region
from src.spatial_index import SpatialIndex
from src.utils.common_utils import Region, Point

PAGE_SIZE = 4096
NODE_SIZE = 1 + 1 + 8 * 4 + 4 * 4 + 4 * 2  # 58
ITEM_SIZE = 8 * 2 + 4  # 20
NODES_PER_PAGE = int(PAGE_SIZE / NODE_SIZE)
ITEMS_PER_PAGE = int(PAGE_SIZE / ITEM_SIZE)


class PRQuadTree(SpatialIndex):
    """
    点分四叉树（Point Range Quadtree，PR Quadtree）
    Implement from the paper The quadtree and related hierarchical data structures
    """

    def __init__(self, model_path=None, root_node=None):
        super(PRQuadTree, self).__init__("PRQuadTree")
        self.root_node = root_node
        self.threshold_number = None
        self.max_depth = None
        self.model_path = model_path
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # for compute
        self.io_cost = 0

    def insert_single(self, point):
        self.insert_node(Point(point[0], point[1], key=point[2]), self.root_node)

    def insert_node(self, point, node):
        """
        插入元素
        1.判断是否已分裂，已分裂的选择适合的子节点，插入；
        2.未分裂的查看过载和树高，过载且树高未满的分裂节点，重新插入；
        3.未过载的直接添加
        :param node:
        :param point:
        """
        if node.is_leaf == 1:
            if len(node.items) >= self.threshold_number and node.depth < self.max_depth:
                self.split_node(node)
                self.insert_node(point, node)
            else:
                node.items.append(point)
            return

        y_center = (node.region.up + node.region.bottom) / 2
        x_center = (node.region.left + node.region.right) / 2
        if point.lat < y_center:
            if point.lng < x_center:
                self.insert_node(point, node.LB)
            else:
                self.insert_node(point, node.RB)
        else:
            if point.lng < x_center:
                self.insert_node(point, node.LU)
            else:
                self.insert_node(point, node.RU)

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
            self.insert_node(item, node)

        # 清空父节点的element
        node.items = []

    def create_child_node(self, node, bottom, up, left, right):
        depth = node.depth + 1
        region = Region(bottom, up, left, right)
        child_node = Node(region=region, depth=depth)
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
                if node.items[i] == point and node.items[i].key == point.key:
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
                if (len(node.RU.items) + len(node.LU.items) + len(node.RB.items) + len(
                        node.LB.items)) <= self.threshold_number:
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
        if node.is_leaf == 1:
            self.io_cost += math.ceil(len(node.items) / ITEMS_PER_PAGE)
            return [item.key for item in node.items if item == point]
        self.io_cost += 1
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
        self.io_cost += 1
        y_center = (node.region.up + node.region.bottom) / 2
        x_center = (node.region.left + node.region.right) / 2
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

    def build(self, data_list, region, threshold_number, data_precision):
        self.threshold_number = threshold_number
        self.max_depth = region.get_max_depth_by_region_and_precision(precision=data_precision)
        self.root_node = Node(region=region)
        self.insert(data_list)

    def point_query_single(self, point):
        """
        1. search by x/y
        2. for duplicate point: only return the first one
        """
        return self.search(Point(point[0], point[1]))

    def range_search_by_iter(self, region, result, node=None):
        node = self.root_node if node is None else node
        if node.region == region:
            tmp_result = []
            node.get_all_items(tmp_result)
            for items in tmp_result:
                self.io_cost += math.ceil(len(items) / ITEMS_PER_PAGE)
                result.extend([item.key for item in items])
            return
        if node.is_leaf == 1:
            self.io_cost += math.ceil(len(node.items) / ITEMS_PER_PAGE)
            result.extend([item.key for item in node.items if region.contain_and_border_by_point(item)])
        else:
            self.io_cost += 1
            # 所有的or：region的四至点刚好在子节点的region上，因为split的时候经纬度都是向上取整，所以子节点的重心在右和上
            if node.LB.region.contain(Point(region.left, region.bottom)):
                self.range_search(Region(region.bottom, min(node.LB.region.up, region.up),
                                         region.left, min(node.LB.region.right, region.right)),
                                  result, node.LB)
            if node.RB.region.contain(Point(region.right, region.bottom)) \
                    or (region.bottom < node.RB.region.up and region.right == node.RB.region.right):
                self.range_search(Region(region.bottom, min(node.LB.region.up, region.up),
                                         max(node.RU.region.left, region.left), region.right),
                                  result, node.RB)
            if node.LU.region.contain(Point(region.left, region.up)) \
                    or (region.left < node.LU.region.right and region.up == node.LU.region.up):
                self.range_search(Region(max(node.RU.region.bottom, region.bottom), region.up,
                                         region.left, min(node.LB.region.right, region.right)),
                                  result, node.LU)
            if node.RU.region.contain(Point(region.right, region.up)) \
                    or (region.right > node.RU.region.left and region.up == node.RU.region.up) \
                    or (region.up > node.RU.region.bottom and region.right == node.RU.region.right):
                self.range_search(Region(max(node.RU.region.bottom, region.bottom), region.up,
                                         max(node.RU.region.left, region.left), region.right),
                                  result, node.RU)

    def range_query_single(self, window):
        result = []
        self.range_search_by_iter(region=Region(window[0], window[1], window[2], window[3]), result=result, node=None)
        return result

    def knn_query_single_t2d(self, knn):
        """
        自上而下
        代码参考：https://github.com/diana12333/QuadtreeNN
        1.用root node初始化stack，nearest_distance和point_heap分别为正无穷和空
        2.循环：当stack非空
        2.1.如果节点距离够：node.region和point的距离不超过nearest_distance
        2.1.1.如果node is_leaf，则遍历items，用距离够的item更新point_heap，同时更新nearest_distance
        2.1.2.如果node not_leaf，则把child放入stack
        3.返回result里的所有key
        """
        point = Point(knn[0], knn[1])
        n = knn[2]
        stack = [self.root_node]
        nearest_distance = (float('-inf'), None)
        point_heap = []
        while len(stack):
            cur = stack.pop(-1)
            if cur.region.within_distance_pow(point, -nearest_distance[0]):
                if cur.is_leaf:
                    self.io_cost += math.ceil(len(cur.items) / ITEMS_PER_PAGE)
                    for item in cur.items:
                        point_distance = point.distance_pow(item)
                        if len(point_heap) < n:
                            heapq.heappush(point_heap, (-point_distance, item.key))
                            nearest_distance = heapq.nsmallest(1, point_heap)[0]
                        elif point_distance < -nearest_distance[0]:
                            heapq.heappop(point_heap)
                            heapq.heappush(point_heap, (-point_distance, item.key))
                            nearest_distance = heapq.nsmallest(1, point_heap)[0]
                else:
                    stack.extend([cur.LB, cur.RB, cur.LU, cur.RU])
        return [itr[1] for itr in point_heap]

    def knn_query_single(self, knn):
        """
        自下而上
        1.先找到point所在的节点，初始化nearest_distance和point_heap
        2.后续操作和自上而下一致，但是由于nearest_distance被初始化，后续遍历可以减少大量节点的距离判断
        检索时间从0.099225优化到0.006712
        待优化: stack改成iter并测试可否进一步加速
        """
        point = Point(knn[0], knn[1])
        n = int(knn[2])
        stack = [self.root_node]
        nearest_distance = (float('-inf'), None)
        point_heap = []
        point_node = self.search_node(point)
        self.io_cost += math.ceil(len(point_node.items) / ITEMS_PER_PAGE)
        for item in point_node.items:
            point_distance = point.distance_pow(item)
            if len(point_heap) < n:
                heapq.heappush(point_heap, (-point_distance, item.key))
                nearest_distance = heapq.nsmallest(1, point_heap)[0]
            elif point_distance < -nearest_distance[0]:
                heapq.heappop(point_heap)
                heapq.heappush(point_heap, (-point_distance, item.key))
                nearest_distance = heapq.nsmallest(1, point_heap)[0]
        while len(stack):
            cur = stack.pop(-1)
            # 跳过point所在节点的判断
            if cur == point_node:
                continue
            if cur.region.within_distance_pow(point, -nearest_distance[0]):
                if cur.is_leaf:
                    self.io_cost += math.ceil(len(cur.items) / ITEMS_PER_PAGE)
                    for item in cur.items:
                        point_distance = point.distance_pow(item)
                        if len(point_heap) < n:
                            heapq.heappush(point_heap, (-point_distance, item.key))
                            nearest_distance = heapq.nsmallest(1, point_heap)[0]
                        elif point_distance < -nearest_distance[0]:
                            heapq.heappop(point_heap)
                            heapq.heappush(point_heap, (-point_distance, item.key))
                            nearest_distance = heapq.nsmallest(1, point_heap)[0]
                elif not cur.is_leaf:
                    stack.extend([cur.LB, cur.RB, cur.LU, cur.RU])
        return [itr[1] for itr in point_heap]

    def save(self):
        """
        以DFS的顺序把tree保存为list
        """
        node_list = []
        item_list = []
        tree_to_list(self.root_node, node_list, item_list)
        prqt_tree = np.array([tuple(node) for node in node_list],
                             dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'), ("3", 'i4'), ("4", 'i4'), ("5", 'i4'),
                                    ("6", 'i4'), ("7", 'i4'), ("8", 'f8'), ("9", 'f8'), ("10", 'f8'), ("11", 'f8')])
        prqt_item = np.array(item_list, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i4')])
        prqt_meta = np.array([self.max_depth, self.threshold_number], dtype=np.int32)
        np.save(os.path.join(self.model_path, 'prquadtree_tree.npy'), prqt_tree)
        np.save(os.path.join(self.model_path, 'prquadtree_item.npy'), prqt_item)
        np.save(os.path.join(self.model_path, 'prquadtree_meta.npy'), prqt_meta)

    def load(self):
        prqt_tree = np.load(os.path.join(self.model_path, 'prquadtree_tree.npy'), allow_pickle=True)
        prqt_item = np.load(os.path.join(self.model_path, 'prquadtree_item.npy'), allow_pickle=True)
        prqt_meta = np.load(os.path.join(self.model_path, 'prquadtree_meta.npy'))
        self.root_node = list_to_tree(prqt_tree, prqt_item)
        self.max_depth = prqt_meta[0]
        self.threshold_number = prqt_meta[1]

    def size(self):
        """
        structure_size = prquadtree_tree.npy + prquadtree_meta.npy
        ie_size = prquadtree_item.npy
        """
        return os.path.getsize(os.path.join(self.model_path, "prquadtree_tree.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "prquadtree_meta.npy")) - 128, \
               os.path.getsize(os.path.join(self.model_path, "prquadtree_item.npy")) - 128


class Node:
    def __init__(self, region, depth=1, is_leaf=1, LB=None, RB=None, LU=None, RU=None, items=None):
        self.depth = depth
        self.is_leaf = is_leaf
        self.region = region
        self.LB = LB
        self.RB = RB
        self.LU = LU
        self.RU = RU
        self.items = items if items else []

    def get_all_items(self, result):
        if self.is_leaf == 1:
            result.append(self.items)
        else:
            self.LB.get_all_items(result)
            self.RB.get_all_items(result)
            self.LU.get_all_items(result)
            self.RU.get_all_items(result)


def tree_to_list(node, node_list, item_list):
    if node is None:
        return
    old_item_len = len(item_list)
    item_list.extend([(item.lng, item.lat, item.key) for item in node.items])
    item_len = len(item_list)
    node_list.append([0, 0, 0, 0, node.depth, node.is_leaf, old_item_len, item_len,
                      node.region.bottom, node.region.up, node.region.left, node.region.right])
    parent_key = len(node_list) - 1
    if node.LB:
        node_list[parent_key][0] = len(node_list)
        tree_to_list(node.LB, node_list, item_list)
    if node.LU is not None:
        node_list[parent_key][1] = len(node_list)
        tree_to_list(node.LU, node_list, item_list)
    if node.RB is not None:
        node_list[parent_key][2] = len(node_list)
        tree_to_list(node.RB, node_list, item_list)
    if node.RU is not None:
        node_list[parent_key][3] = len(node_list)
        tree_to_list(node.RU, node_list, item_list)


def list_to_tree(node_list, item_list, key=None):
    if key is None:
        key = 0
    item = node_list[key]
    region = Region(item[8], item[9], item[10], item[11])
    items = [Point(point[0], point[1], key=point[2]) for point in item_list[item[6]:item[7]]]
    node = Node(region, item[4], item[5], None, None, None, None, items)
    if item[0]:
        node.LB = list_to_tree(node_list, item_list, item[0])
    if item[1]:
        node.LU = list_to_tree(node_list, item_list, item[1])
    if item[2]:
        node.RB = list_to_tree(node_list, item_list, item[2])
    if item[3]:
        node.RU = list_to_tree(node_list, item_list, item[3])
    return node


def get_leaf_and_path(node_list, result, cur_path, key):
    item = node_list[key]
    cur_path.append(key)
    if item[5]:
        result.append([cur_path, item[7] - item[6]])
        return
    else:
        get_leaf_and_path(node_list, result, copy.deepcopy(cur_path), item[0])
        get_leaf_and_path(node_list, result, copy.deepcopy(cur_path), item[1])
        get_leaf_and_path(node_list, result, copy.deepcopy(cur_path), item[2])
        get_leaf_and_path(node_list, result, copy.deepcopy(cur_path), item[3])


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/prquadtree_10w/"
    data_distribution = Distribution.NYCT_10W
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = PRQuadTree(model_path=model_path)
    index_name = index.name
    load_index_from_file = False
    if load_index_from_file:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(data_distribution, 0)
        index.build(data_list=build_data_list,
                    threshold_number=1000,
                    data_precision=data_precision[data_distribution],
                    region=data_region[data_distribution])
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    structure_size, ie_size = index.size()
    logging.info("Structure size: %s" % structure_size)
    logging.info("Index entry size: %s" % ie_size)
    io_cost = 0
    path = '../../data/query/point_query_nyct.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    range_query_list = load_query(data_distribution, 1).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(range_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    knn_query_list = load_query(data_distribution, 2).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(knn_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Update time: %s" % (end_time - start_time))
    logging.info("Update io cost: %s" % (index.io_cost - io_cost))
    io_cost = index.io_cost
    point_query_list = load_query(data_distribution, 0).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    logging.info("Point query io cost: %s" % ((index.io_cost - io_cost) / len(point_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'point_query_result1.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
