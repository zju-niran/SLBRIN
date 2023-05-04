import gc
import heapq
import logging
import os
import time

import numpy as np

from src.experiment.common_utils import Distribution, load_data
from src.spatial_index import SpatialIndex

DIM_NUM = 2
PAGE_SIZE = 4096
NODE_SIZE = 1 + 4 + 4 * 2 + (8 * 2 + 4)  # 33
NODES_PER_RA = int(PAGE_SIZE / NODE_SIZE)


class KDTree(SpatialIndex):
    """
    KD树（KD-tree）
    Implement from Multidimensional binary search trees used for associative searching
    """

    def __init__(self, model_path=None):
        super(KDTree, self).__init__("KDTree")
        self.model_path = model_path
        self.root_node = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        # for compute
        self.io_cost = 0

    def insert_single(self, point):
        self.root_node = self.root_node.insert((point[0], point[1], point[3]))

    def insert(self, points):
        for point in points:
            self.insert_single(point)
        # balance: 将树调整回平衡状态，检索效率会提升
        # 理论上要在insert和delete时处理最后被更新的node和所有上层node，但是balance太耗时，此处旨在整体insert是操作一次
        self.root_node.balance()

    def delete(self, point):
        self.root_node = self.root_node.delete(point)

    def build_node(self, values, value_len, axis):
        """
        通过del减少内存占用
        """
        median_key = value_len // 2
        values.sort(key=lambda x: x[axis])
        node = KDNode(value=values[median_key], axis=axis)
        left_value_len = median_key
        right_value_len = value_len - median_key - 1
        axis = axis + 1 if axis + 1 < DIM_NUM else 0
        values_left = values[:median_key]
        values_right = values[median_key + 1:]
        del values
        if left_value_len > 0:
            node.left = self.build_node(values_left, left_value_len, axis)
        del values_left
        if right_value_len > 0:
            node.right = self.build_node(values_right, right_value_len, axis)
        del values_right
        node.recalculate_nodes()
        gc.collect(generation=0)
        return node

    def build(self, data_list):
        # 方法1：先排序后划分节点
        data_list = [(data[0], data[1], data[3]) for data in data_list]
        self.root_node = self.build_node(data_list, len(data_list), 0)
        # 方法2：不停插入
        # data_list = np.insert(data_list, np.arange(len(data_list)), axis=1)
        # self.root_node = KDNode(value=data_list[0], axis=0)
        # self.insert(data_list[1:])
        self.root_node.balance()
        # self.visualize("1.txt")

    def visualize(self, output_path):
        result = []
        self.root_node.visualize(result=result)
        with open(os.path.join(self.model_path, output_path), 'w') as f:
            for line in result:
                f.write(line + '\r')

    def point_query_single(self, point):
        result = []
        self.point_query_node(self.root_node, point, result)
        return result

    def point_query_node(self, node, point, result):
        if point[node.axis] > node.value[node.axis]:
            if node.right:
                self.point_query_node(node.right, point, result)
        elif point[node.axis] == node.value[node.axis]:
            if equal_value_2d(point, node.value):
                result.append(node.value[-1])
            if node.right:
                self.point_query_node(node.right, point, result)
            if node.left:
                self.point_query_node(node.left, point, result)
        else:
            if node.left:
                self.point_query_node(node.left, point, result)

    def range_query_single(self, window):
        """
        优化：stack->当前方法:2->1
        """
        result = []
        window = [window[2], window[3], window[0], window[1]]
        self.range_query_node(self.root_node, window, result)
        return result

    def range_query_node(self, node, window, result):
        if contain_value_2d(window, node.value):
            result.append(node.value[-1])
        if node.left:
            if window[node.axis * 2] <= node.value[node.axis]:
                self.range_query_node(node.left, window, result)
        if node.right:
            if window[node.axis * 2 + 1] >= node.value[node.axis]:
                self.range_query_node(node.right, window, result)

    def range_query_by_stack(self, window):
        result = []
        stack = [self.root_node]
        window = [window[2], window[3], window[0], window[1]]
        while len(stack):
            cur = stack.pop(-1)
            if contain_value_2d(window, cur.value):
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
        参考：https://blog.csdn.net/qq_38019633/article/details/89555909
        1. 从root node开始深度优先遍历
        2. 先判断当前node value和value的大小关系，如果=大，则顺序为right-自己-left，否则left-自己-right
        3. 对前节点，直接遍历
        4. 对自己，直接加进优先队列，并且更新距离
        5. 对后节点，可以借助和node的分割线的距离判断，加速过滤
        优化：stack->iter->当前方法:时间比为50->50->0.1
        """
        value = knn[:-1]
        n = int(knn[-1])
        result_heap = []
        self.knn_query_node(self.root_node, value, 0, result_heap, n)
        return [itr[1] for itr in result_heap]

    def knn_query_node(self, node, value, nearest_distance, result_heap, n):
        # 右-自己-左
        if value[node.axis] >= node.value[node.axis]:
            if node.right:
                nearest_distance = self.knn_query_node(node.right, value, nearest_distance, result_heap, n)
            dst = distance_value_2d(value, node.value)
            if len(result_heap) < n:
                heapq.heappush(result_heap, (-dst, node.value[2]))
                nearest_distance = -heapq.nsmallest(1, result_heap)[0][0]
            elif dst < nearest_distance:
                heapq.heappop(result_heap)
                heapq.heappush(result_heap, (-dst, node.value[2]))
                nearest_distance = -heapq.nsmallest(1, result_heap)[0][0]
            if node.left and value[node.axis] - nearest_distance < node.value[node.axis]:
                nearest_distance = self.knn_query_node(node.left, value, nearest_distance, result_heap, n)
        else:  # 左-自己-右
            if node.left:
                nearest_distance = self.knn_query_node(node.left, value, nearest_distance, result_heap, n)
            dst = distance_value_2d(value, node.value)
            if len(result_heap) < n:
                heapq.heappush(result_heap, (-dst, node.value[2]))
                nearest_distance = -heapq.nsmallest(1, result_heap)[0][0]
            elif dst < nearest_distance:
                heapq.heappop(result_heap)
                heapq.heappush(result_heap, (-dst, node.value[2]))
                nearest_distance = -heapq.nsmallest(1, result_heap)[0][0]
            if node.right and value[node.axis] + nearest_distance >= node.value[node.axis]:
                nearest_distance = self.knn_query_node(node.right, value, nearest_distance, result_heap, n)
        return nearest_distance

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

    def save(self):
        node_list = []
        tree_to_list(self.root_node, node_list)
        kd_tree = np.array(node_list, dtype=[("0", 'i4'), ("1", 'i4'), ("2", 'i1'), ("3", 'i4'),
                                             ("4", 'f8'), ("5", 'f8'), ("6", 'i4')])
        np.save(os.path.join(self.model_path, 'kd_tree.npy'), kd_tree)

    def load(self):
        kd_tree = np.load(os.path.join(self.model_path, 'kd_tree.npy'), allow_pickle=True)
        self.root_node = list_to_tree(kd_tree, 0)

    def size(self):
        """
        ie_size = data_len * data_size
        structure_size = kd_tree.npy - ie_size
        """
        size = os.path.getsize(os.path.join(self.model_path, "kd_tree.npy")) - 128 - 64
        data_len = self.root_node.node_num
        data_size = 20
        ie_size = data_len * data_size
        structure_size = size - ie_size
        return structure_size, ie_size


class KDNode:

    def __init__(self, value=None, axis=0):
        self.value = value
        self.left = None
        self.right = None
        self.axis = axis
        self.node_num = 1

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
        self.recalculate_nodes()
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
                self.recalculate_nodes()
                return self
        else:
            if self.left is None:
                return self
            else:
                self.left = self.left.delete(value)
                self.recalculate_nodes()
                return self

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

    def recalculate_nodes(self):
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
        node.recalculate_nodes()
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


def distance_value(value1, value2):
    return sum([(value1[d] - value2[d]) ** 2 for d in range(DIM_NUM)]) ** 0.5


def equal_value(value1, value2):
    return sum([value1[d] == value2[d] for d in range(DIM_NUM)]) == DIM_NUM


def contain_value(window, value):
    return sum([window[d * 2] <= value[d] <= window[d * 2 + 1] for d in range(DIM_NUM)]) == DIM_NUM


# 2d下的函数效率要比多维快5倍
def equal_value_2d(value1, value2):
    return value1[0] == value2[0] and value1[1] == value2[1]


def distance_value_2d(value1, value2):
    return ((value1[0] - value2[0]) ** 2 + (value1[1] - value2[1]) ** 2) ** 0.5


def contain_value_2d(window, value):
    return window[0] <= value[0] <= window[1] and window[2] <= value[1] <= window[3]


def tree_to_list(node, node_list):
    if node is None:
        return
    node_list.append((0, 0, node.axis, node.node_num, node.value[0], node.value[1], node.value[2]))
    parent_key = len(node_list) - 1
    if node.left:
        node_list[parent_key] = (len(node_list),) + node_list[parent_key][1:]
        tree_to_list(node.left, node_list)
    if node.right is not None:
        node_list[parent_key] = node_list[parent_key][:1] + (len(node_list),) + node_list[parent_key][2:]
        tree_to_list(node.right, node_list)


def list_to_tree(node_list, key):
    item = list(node_list[key])
    node = KDNode(item[4:], item[2])
    node.node_num = item[3]
    if item[0]:
        node.left = list_to_tree(node_list, item[0])
    if item[1]:
        node.right = list_to_tree(node_list, item[1])
    return node


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    model_path = "model/kdtree_10w/"
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    index = KDTree(model_path=model_path)
    index_name = index.name
    load_index_from_file = True
    if load_index_from_file:
        index.load()
    else:
        index.logging.info("*************start %s************" % index_name)
        start_time = time.time()
        build_data_list = load_data(Distribution.NYCT_10W, 0)
        # 按照pagesize=4096, read_ahead=256, size(pointer)=4, size(x/y)=8, node按照DFS的顺序密集存储在page中
        # tree存放所有node的axis、数据量、左右节点指针、data:
        # node size=1+4+4*2+(8*2+4)=33，单page存放4096/33=124node，单read_ahead读取256*124=31744node
        # 15层节点数=2^(15-1)=16384，之后每一层对应1次IO
        # 10w数据，[]参数下：
        # 树高=log2(10w)=17, IO=前15层IO+后17-15层IO=1~2+2=3~4
        # 索引体积=33*10w
        # 1451w数据，[]参数下：
        # 树高=log2(1451W)=24, IO=前15层IO+后24-15层IO=1~2+9=10~11
        # 索引体积=33*1451w
        index.build(data_list=build_data_list)
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
    path = '../../data/query/range_query_nyct.npy'
    range_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.range_query(range_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(range_query_list)
    logging.info("Range query time: %s" % search_time)
    logging.info("Range query io cost: %s" % ((index.io_cost - io_cost) / len(range_query_list)))
    io_cost = index.io_cost
    np.savetxt(model_path + 'range_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    path = '../../data/query/knn_query_nyct.npy'
    knn_query_list = np.load(path, allow_pickle=True).tolist()
    start_time = time.time()
    results = index.knn_query(knn_query_list)
    end_time = time.time()
    search_time = (end_time - start_time) / len(knn_query_list)
    logging.info("KNN query time: %s" % search_time)
    logging.info("KNN query io cost: %s" % ((index.io_cost - io_cost) / len(knn_query_list)))
    np.savetxt(model_path + 'knn_query_result.csv', np.array(results, dtype=object), delimiter=',', fmt='%s')
    update_data_list = load_data(Distribution.NYCT_10W, 1)
    start_time = time.time()
    index.insert(update_data_list)
    end_time = time.time()
    logging.info("Update time: %s" % (end_time - start_time))


if __name__ == '__main__':
    main()
