import heapq
import logging
import os
import sys
import time

import numpy as np

sys.path.append('/home/zju/wlj/st-learned-index')
from src.spatial_index.spatial_index import SpatialIndex

"""
二维的话直接用bool来处理维度切换性能会高，现在axis不停+1来迭代
"""
DIM_NUM = 2


def distance_value(value1, value2):
    return sum([(value1[d] - value2[d]) ** 2 for d in range(DIM_NUM)]) ** 0.5


def equal_value(value1, value2):
    return sum([value1[d] == value2[d] for d in range(DIM_NUM)]) == DIM_NUM


def contain_value(window, value):
    return sum([window[d * 2] <= value[d] <= window[d * 2 + 1] for d in range(DIM_NUM)]) == DIM_NUM


class KDNode:
    def __init__(self, value=None, axis=0):
        self.value = value
        self.left = None
        self.right = None
        self.axis = axis
        self.node_num = 1

    def search_node(self, value):
        """
        Search the node for a value
        """
        if value[self.axis] > self.value[self.axis]:
            if self.right is None:
                return self.right
            else:
                return self.right.search_node(value)
        elif value[self.axis] == self.value[self.axis]:
            return self
        else:
            if self.left is None:
                return self.left
            else:
                return self.left.search_node(value)

    def search_all(self, value, result):
        """
        Search all value under the node
        """
        if equal_value(self.value, value):
            result.append(self.value[-1])
        if self.left:
            self.left.search_all(value, result)
        if self.right:
            self.right.search_all(value, result)

    def search_linked_node(self, value, result):
        """
        Search the node for a value and record all the accessed node
        """
        if value[self.axis] > self.value[self.axis]:
            if self.right is None:
                result.append([self, None])
            else:
                result.append([self, False])
                self.right.search_linked_node(value, result)
        elif value[self.axis] == self.value[self.axis]:
            result.append([self, None])
        else:
            if self.left is None:
                result.append([self, None])
            else:
                result.append([self, True])
                self.left.search_linked_node(value, result)

    def nearest_neighbor(self, value, n, result, nearest_distance):
        """
        Determine the `n` nearest node to `value` and their distances.
        eg: self.nearest_neighbor(value, 4, result, float('-inf'))
        """
        dist = distance_value(value, self.value)
        if len(result) < n:
            heapq.heappush(result, (-dist, self.value[-1]))
            nearest_distance = -heapq.nsmallest(1, result)[0][0]
        elif dist < nearest_distance:
            heapq.heappop(result)
            heapq.heappush(result, (-dist, self.value[-1]))
            nearest_distance = -heapq.nsmallest(1, result)[0][0]
        if value[self.axis] + nearest_distance >= self.value[self.axis] and self.right:
            nearest_distance = self.right.nearest_neighbor(value, n, result, nearest_distance)
        if value[self.axis] - nearest_distance < self.value[self.axis] and self.left:
            nearest_distance = self.left.nearest_neighbor(value, n, result, nearest_distance)
        return nearest_distance

    def insert(self, value):
        """
        Insert a value into the node.
        """
        # 重复数据处理：插入时先放到右侧
        if value[self.axis] >= self.value[self.axis]:
            if self.right is None:
                axis = self.axis + 1 if self.axis + 1 < DIM_NUM else 0
                self.right = KDNode(value=value, axis=axis)
            else:
                self.right = self.right.insert(value)
        else:
            if self.left is None:
                axis = self.axis + 1 if self.axis + 1 < DIM_NUM else 0
                self.left = KDNode(value=value, axis=axis)
            else:
                self.left = self.left.insert(value)
        self._recalculate_nodes()
        return self

    def delete(self, value):
        """
        Delete a value from the node and return the new node.
        Returns the same tree if the value was not found.
        """
        if np.all(self.value == value):
            values = self.collect()
            if len(values) > 1:
                values.remove(value)
                new_tree = KDNode.initialize(values, init_axis=self.axis)
                return new_tree
            return None
        elif value[self.axis] >= self.value[self.axis]:
            if self.right is None:
                return self
            else:
                self.right = self.right.delete(value)
                self._recalculate_nodes()
                return self.balance()
        else:
            if self.left is None:
                return self
            else:
                self.left = self.left.delete(value)
                self._recalculate_nodes()
                return self.balance()

    def balance(self):
        """
        Balance the node if the secondary invariant is not satisfied.
        """
        if not self.invariant():
            values = self.collect()
            return KDNode.initialize(values, init_axis=self.axis)
        return self

    def invariant(self):
        """
        Verify that the node satisfies the secondary invariant.
        """
        ln, rn = 0, 0
        if self.left:
            ln = self.left.node_num
        if self.right:
            rn = self.right.node_num
        return abs(ln - rn) <= DIM_NUM

    def _recalculate_nodes(self):
        """
        Recalculate the number of nodes of the node, assuming that the node's children are correctly calculated.
        """
        node_num = 0
        if self.right:
            node_num += self.right.node_num
        if self.left:
            node_num += self.left.node_num
        self.node_num = node_num + 1

    def collect(self):
        """
        Collect all values in the node as a list, ordered in a depth-first manner.
        """
        values = [self.value]
        if self.right:
            values += self.right.collect()
        if self.left:
            values += self.left.collect()
        return values

    def collect_distance(self, value):
        """
        Collect all distance from specified value and all the value under the node
        """
        distances = [[distance_value(value, self.value), self.value[-1]]]
        if self.right:
            distances.extend(self.right.collect_distance(value))
        if self.left:
            distances.extend(self.left.collect_distance(value))
        return distances

    @staticmethod
    def _initialize_recursive(sorted_values, axis):
        """
        Internal recursive initialization based on an array of values presorted in all axes.
        This function should not be called externally. Use `initialize` instead.
        """
        value_len = len(sorted_values[axis])
        median = value_len // 2
        median_value = sorted_values[axis][median]
        median_mask = np.equal(sorted_values[:, :, 2], sorted_values[axis][median][2])
        sorted_values = sorted_values[~median_mask].reshape((DIM_NUM, value_len - 1, 3))
        node = KDNode(median_value, axis=axis)
        right_values = sorted_values[axis][median:]
        right_value_len = len(right_values)
        left_value_len = value_len - 1 - right_value_len
        right_mask = np.isin(sorted_values[:, :, 2], right_values[:, 2])
        sorted_right_values = sorted_values[right_mask].reshape((DIM_NUM, right_value_len, 3))
        sorted_left_values = sorted_values[~right_mask].reshape((DIM_NUM, left_value_len, 3))
        axis = axis + 1 if axis + 1 < DIM_NUM else 0
        if right_value_len > 0:
            node.right = KDNode._initialize_recursive(sorted_right_values, axis)
        if left_value_len > 0:
            node.left = KDNode._initialize_recursive(sorted_left_values, axis)
        node._recalculate_nodes()
        return node

    @staticmethod
    def initialize(values, init_axis=0):
        """
        Initialize a node from a list of values by presorting `values` by each of the axes of discrimination.
        Initialization attempts balancing by selecting the median along each axis of discrimination as the root.
        """
        sorted_values = []
        for axis in range(DIM_NUM):
            sorted_values.append(sorted(values, key=lambda x: x[axis]))
        return KDNode._initialize_recursive(np.asarray(sorted_values), init_axis)

    def visualize(self, depth=0, result=None):
        """
        Prints a visual representation of the KDTree.
        """
        result.append("value: %s, depth: %s, axis: %s, node_num: %s" % (self.value, depth, self.axis, self.node_num))
        if self.left:
            self.left.visualize(depth=depth + 1, result=result)
        if self.right:
            self.right.visualize(depth=depth + 1, result=result)


class KDTree(SpatialIndex):
    def __init__(self, model_path=None):
        super(KDTree, self).__init__("KDTree")
        self.model_path = model_path
        self.root_node = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    def insert(self, point):
        self.root_node = self.root_node.insert(point)

    def insert_batch(self, points):
        for i in range(len(points)):
            self.insert([points[i][0], points[i][1], i])

    def delete(self, point):
        self.index.delete(point.key, (point.lng, point.lat))

    def build_node(self, values, value_len, axis):
        median_key = value_len // 2
        sorted_value = sorted(values, key=lambda x: x[axis])
        node = KDNode(value=sorted_value[median_key], axis=axis)
        left_value_len = median_key
        right_value_len = value_len - median_key - 1
        axis = axis + 1 if axis + 1 < DIM_NUM else 0
        if left_value_len > 0:
            node.left = self.build_node(sorted_value[: median_key], left_value_len, axis)
        if right_value_len > 0:
            node.right = self.build_node(sorted_value[median_key + 1:], right_value_len, axis)
        return node

    def build(self, data_list):
        # TODO: 对比一下两种build的结果树是否一样
        self.root_node = self.build_node(data_list.tolist(), len(data_list), 0)
        # data_list = np.insert(data_list, np.arange(len(data_list)), axis=1)
        # self.root_node = KDNode(value=data_list[0], axis=0)
        # self.insert_batch(data_list[1:])
        # self.root_node.balance()
        # self.visualize("1.txt")

    def visualize(self, output_path):
        result = []
        self.root_node.visualize(result=result)
        with open(os.path.join(self.model_path, output_path), 'w') as f:
            for line in result:
                f.write(line + '\r')

    def point_query_single(self, point):
        result = []
        node = self.root_node.search_node(point)
        if node is None:
            point("")
        node.search_all(point, result)
        return result

    def range_query_single(self, window):
        result = []
        stack = [self.root_node]
        window = [window[2], window[3], window[0], window[1]]
        while len(stack):
            cur = stack.pop(-1)
            if contain_value(window, cur.value):
                result.append(cur.value[-1])
            if cur.left:
                if window[cur.axis * 2] <= cur.value[cur.axis]:
                    stack.append(cur.left)
            if cur.right:
                if window[cur.axis * 2 + 1] >= cur.value[cur.axis]:
                    stack.append(cur.right)
        return result

    def knn_query_single(self, knn):
        """
        1. search the node containing value and record the search link
        2. search node from bottom to up according to the search link
        2.1 if result is not full or current node is closer than nearest_dist,
            collect all the values under current node to result and update nearest_dist
        对比stack/iter/当前方法=>时间比为50:50:0.3
        """
        value = knn
        n = knn[-1]
        # 1. search the node containing value and record the search link
        linked_nodes = []
        self.root_node.search_linked_node(value, linked_nodes)
        nearest_distance = 0
        result = []
        # 2. search node from bottom to up according to the search link
        while len(linked_nodes):
            node, is_left = linked_nodes.pop(-1)
            if is_left is False:
                rest_node = node.left
            elif is_left is True:
                rest_node = node.right
            else:
                rest_node = node
            if rest_node is None:
                continue
            if len(result) >= n:
                if is_left is True and value[node.axis] + nearest_distance < rest_node.value[node.axis]:
                    continue
                elif is_left is False and value[node.axis] - nearest_distance > rest_node.value[node.axis]:
                    continue
            result.extend(rest_node.collect_distance(value))
            result = sorted(result)[:n]
            nearest_distance = result[-1][0]
        return [itr[1] for itr in result]

    def knn_query_by_iter(self, knn):
        result = []
        self.root_node.nearest_neighbor(knn, knn[-1], result, float('-inf'))
        return [itr[1] for itr in result]

    def knn_query_by_stack(self, knn):
        nearest_distance = float('-inf')
        result = []
        n = knn[-1]
        value = knn
        stack = [self.root_node]
        while len(stack):
            cur = stack.pop(-1)
            dist = distance_value(cur.value, value)
            if len(result) < n:
                heapq.heappush(result, (-dist, cur.value[-1]))
                nearest_distance = -heapq.nsmallest(1, result)[0][0]
            elif dist < nearest_distance:
                heapq.heappop(result)
                heapq.heappush(result, (-dist, cur.value[-1]))
                nearest_distance = -heapq.nsmallest(1, result)[0][0]
            if cur.right and value[cur.axis] + nearest_distance >= cur.value[cur.axis]:
                stack.append(cur.right)
            if cur.left and value[cur.axis] - nearest_distance < cur.value[cur.axis]:
                stack.append(cur.left)
        return [itr[1] for itr in result]

    def tree_to_list(self, node, node_list):
        if node is None:
            return
        node_list.append([0, 0, node.axis, node.node_num, node.value[0], node.value[1], node.value[2]])
        parent_key = len(node_list) - 1
        if node.left:
            node_list[parent_key][0] = len(node_list)
            self.tree_to_list(node.left, node_list)
        if node.right is not None:
            node_list[parent_key][1] = len(node_list)
            self.tree_to_list(node.right, node_list)

    def list_to_tree(self, node_list, key=None):
        if key is None:
            key = 0
        item = list(node_list[key])
        node = KDNode(item[4:], item[2])
        if item[0] != 0:
            node.left = self.list_to_tree(node_list, item[0])
        if item[1] != 0:
            node.right = self.list_to_tree(node_list, item[1])
        return node

    def save(self):
        node_list = []
        self.tree_to_list(self.root_node, node_list)
        result1 = []
        self.root_node.search_linked_node([-73.954468, 40.779896], result1)
        kd_tree = np.array([tuple(node) for node in node_list],
                           dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i4'), ("3", 'i4'),
                                  ("4", 'f8'), ("5", 'f8'), ("6", 'i4')])
        np.save(os.path.join(self.model_path, 'kd_tree.npy'), kd_tree)

    def load(self):
        kd_tree = np.load(self.model_path + 'kd_tree.npy', allow_pickle=True)
        self.root_node = self.list_to_tree(kd_tree)

    def size(self):
        """
        size = kd_tree.npy
        """
        return os.path.getsize(os.path.join(self.model_path, "kd_tree.npy")) - 128


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_path = '../../data/table/trip_data_1_filter_10w.npy'
    model_path = "model/kdtree_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = KDTree(model_path=model_path)
    index_name = index.name
    load_index_from_json = True
    if load_index_from_json:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        data_list = np.load(data_path, allow_pickle=True)[:, [10, 11, -1]]
        # 按照pagesize=4096, prefetch=256, size(pointer)=4, size(x/y)=8, node按照DFS的顺序密集存储在page中
        # tree存放所有node的axis、数据量、左右节点指针、data的data指针:
        # node size=4+4+4+4+8*2+4=36，单page存放4096/36=113node，单prefetch读取256*113=26668node
        # 15层节点数=2^(15-1)=16384，之后每一层对应1次IO
        # 10w数据，[]参数下：
        # 树高=log2(10w)=17, IO=前15层IO+后17-15层IO=1~2+2=3~4
        # 索引体积=36*10w
        # 1451w数据，[]参数下：
        # 树高=log2(1451W)=24, IO=前15层IO+后24-15层IO=1~2+9=10~11
        # 索引体积=36*1451w
        index.build(data_list=data_list)
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
