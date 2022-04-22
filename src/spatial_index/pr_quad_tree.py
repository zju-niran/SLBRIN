import heapq
import logging
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.spatial_index import SpatialIndex
from src.spatial_index.common_utils import Region, Point


class QuadTreeNode:
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
            result.extend([item.key for item in self.items])
        else:
            self.LB.get_all_items(result)
            self.RB.get_all_items(result)
            self.LU.get_all_items(result)
            self.RU.get_all_items(result)


class PRQuadTree(SpatialIndex):
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

    def insert(self, point):
        self.insert_node(Point(point[0], point[1], key=point[2]), self.root_node)

    def insert_batch(self, points):
        for point in points:
            self.insert(point)

    def insert_node(self, point, node):
        """
        插入元素
        1.判断是否已分裂，已分裂的选择适合的子节点，插入；
        2.未分裂的查看过载和树高，过载且树高未满的分裂节点，重新插入；
        3.未过载的直接添加
        :param node:
        :param point:
        todo 使用元素原地址，避免重新分配内存造成的效率浪费
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
            return [item.key for item in node.items if item == point]
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
        self.root_node = QuadTreeNode(region=region)
        self.insert_batch(data_list)

    def point_query_single(self, point):
        """
        1. search by x/y
        2. for duplicate point: only return the first one
        """
        return self.search(Point(point[0], point[1]))

    def range_search(self, region, result, node=None):
        node = self.root_node if node is None else node
        if node.region == region:
            node.get_all_items(result)
            return
        if node.is_leaf == 1:
            result.extend([item.key for item in node.items if region.contain_and_border_by_point(item)])
        else:
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
        self.range_search(region=Region(window[0], window[1], window[2], window[3]), result=result, node=None)
        return result

    def knn_query_single_old(self, knn):
        """
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
        1.先找到point所在的节点，初始化nearest_distance和point_heap
        2.后续操作和knn_query_old一致，但是由于nearest_distance被初始化，后续遍历可以减少大量节点的距离判断
        检索时间从0.099225优化到0.006712
        TODO: stack改成iter看下是否有加速
        """
        point = Point(knn[0], knn[1])
        n = knn[2]
        stack = [self.root_node]
        nearest_distance = (float('-inf'), None)
        point_heap = []
        point_node = self.search_node(point)
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

    def tree_to_list(self, node, node_list, item_list):
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
            self.tree_to_list(node.LB, node_list, item_list)
        if node.LU is not None:
            node_list[parent_key][1] = len(node_list)
            self.tree_to_list(node.LU, node_list, item_list)
        if node.RB is not None:
            node_list[parent_key][2] = len(node_list)
            self.tree_to_list(node.RB, node_list, item_list)
        if node.RU is not None:
            node_list[parent_key][3] = len(node_list)
            self.tree_to_list(node.RU, node_list, item_list)

    def list_to_tree(self, node_list, item_list, key=None):
        if key is None:
            key = 0
        item = node_list[key]
        region = Region(item[8], item[9], item[10], item[11])
        items = [Point(point[0], point[1], key=point[2]) for point in item_list[item[6]:item[7]]]
        node = QuadTreeNode(region, item[4], item[5], None, None, None, None, items)
        if item[0] != 0:
            node.LB = self.list_to_tree(node_list, item_list, item[0])
        if item[1] != 0:
            node.LU = self.list_to_tree(node_list, item_list, item[1])
        if item[2] != 0:
            node.RB = self.list_to_tree(node_list, item_list, item[2])
        if item[3] != 0:
            node.RU = self.list_to_tree(node_list, item_list, item[3])
        return node

    def save(self):
        node_list = []
        item_list = []
        self.tree_to_list(self.root_node, node_list, item_list)
        prqt_tree = np.array([tuple(node) for node in node_list],
                             dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'), ("3", 'i4'), ("4", 'i4'), ("5", 'i4'),
                                    ("6", 'i4'), ("7", 'i4'), ("8", 'f8'), ("9", 'f8'), ("10", 'f8'), ("11", 'f8')])
        prqt_item = np.array(item_list, dtype=[("0", 'f8'), ("1", 'f8'), ("2", 'i4')])
        prqt_meta = np.array([self.max_depth, self.threshold_number], dtype=np.int32)
        np.save(os.path.join(self.model_path, 'prquadtree_tree.npy'), prqt_tree)
        np.save(os.path.join(self.model_path, 'prquadtree_item.npy'), prqt_item)
        np.save(os.path.join(self.model_path, 'prquadtree_meta.npy'), prqt_meta)

    def load(self):
        prqt_tree = np.load(self.model_path + 'prquadtree_tree.npy', allow_pickle=True)
        prqt_item = np.load(self.model_path + 'prquadtree_item.npy', allow_pickle=True)
        prqt_meta = np.load(self.model_path + 'prquadtree_meta.npy')
        self.root_node = self.list_to_tree(prqt_tree, prqt_item)
        self.max_depth = prqt_meta[0]
        self.threshold_number = prqt_meta[1]

    def size(self):
        """
        size = prquadtree_tree.npy + prquadtree_item.npy + prquadtree_meta.npy
        """
        return os.path.getsize(os.path.join(self.model_path, "prquadtree_tree.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "prquadtree_item.npy")) - 128 + \
               os.path.getsize(os.path.join(self.model_path, "prquadtree_meta.npy")) - 128


# @profile(precision=8)
def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/table/trip_data_1_filter_10w.npy'
    model_path = "model/prquadtree_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = PRQuadTree(model_path=model_path)
    index_name = index.name
    load_index_from_json = False
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)[:, [10, 11, -1]]
        # 按照pagesize=4096, prefetch=256, size(pointer)=4, size(x/y)=8, node和data按照DFS的顺序密集存储在page中
        # tree存放所有node的深度、是否叶节点、region、四节点指针和data的始末指针:
        # node size=4+4+8*4+4*4+4*2=64，单page存放4096/64=64node，单prefetch读取256*64=16384node
        # item存放xy数据和数据指针：
        # data size=8*2+4=20，单page存放4096/20=204data，单prefetch读取256*204=52224data
        # 10w数据，[1000]参数下：
        # 叶节点平均数据约为0.5*1000=500，叶节点约有10w/500=200个，非叶节点数量由数据分布决定，节点大约280个
        # 单次扫描IO=读取node+读取node对应数据=280/16384+500/16384=2
        # 索引体积=280/64*4096+20*10w
        # 1451w数据，[1000]参数下：
        # 叶节点平均数据约为0.5*1000=500，叶节点约有1451w/500=29020个，非叶节点数量由数据分布决定，节点大约5w个
        # 单次扫描IO=读取node+读取node对应数据=5w/16384+500/16384=4~5
        # 索引体积=5w/64*4096+20*1451w
        index.build(data_list=data_list,
                    region=Region(40, 42, -75, -73),
                    threshold_number=1000,
                    data_precision=6)
        index.save()
        end_time = time.time()
        build_time = end_time - start_time
        index.logging.info("Build time: %s" % build_time)
    logging.info("Index size: %s" % index.size())
    path = '../../data/query/point_query_10w.npy'
    point_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.point_query(point_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(point_query_list)
    logging.info("Point query time: %s" % search_time)
    np.savetxt(model_path + 'point_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/range_query_10w.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time:  %s" % search_time)
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/knn_query_10w.npy'
    knn_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time:  %s" % search_time)
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    insert_data_list = np.load("../../data/table/trip_data_2_filter_10w.npy", allow_pickle=True)[:, [10, 11, -1]]
    start_time = time.time()
    index.insert_batch(insert_data_list)
    end_time = time.time()
    logging.info("Insert time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()