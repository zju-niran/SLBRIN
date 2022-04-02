import heapq
import os

import line_profiler
import numpy as np
import pandas as pd

from src.b_tree import BTree, Item
from src.spatial_index.common_utils import binary_search, binary_search_less_max

if __name__ == '__main__':
    # os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # data = pd.read_csv("./spatial_index/3.csv", header=None)
    # i = 0
    # j = 0
    # model_path = "models/" + str(i) + "_" + str(j) + "_weights.best.hdf5"
    # inputs = data[1].values
    # labels = data[2].values
    # tmp_index = TrainedNN(model_path, inputs, labels,
    #                       0.5,
    #                       True,
    #                       [1, 16, 1],
    #                       30000,
    #                       1024,
    #                       0.01,
    #                       0.9,
    #                       0)
    # tmp_index.train()
    # # get parameters in model (weight matrix and bias matrix)
    # abstract_index = AbstractNN(tmp_index.get_weights(),
    #                             [1, 128, 1],
    #                             tmp_index.train_x_min,
    #                             tmp_index.train_x_max,
    #                             tmp_index.train_y_min,
    #                             tmp_index.train_y_max,
    #                             tmp_index.min_err,
    #                             tmp_index.max_err)
    # pres = abstract_index.predict(inputs)
    # print("adasd")
    def fun1():
        point_heap = [(-1, 1), (-2, 1), (-4, 1), (-3, 1)]
        n = 7
        nearest_distance = (float('-inf'), None)
        for i in b:
            if len(point_heap) < n:
                heapq.heappush(point_heap, (i, 1))
                nearest_distance = heapq.nsmallest(1, point_heap)[0]
                continue
            if i < -nearest_distance[0]:
                heapq.heappop(point_heap)
                heapq.heappush(point_heap, (i, 1))
                nearest_distance = heapq.nsmallest(1, point_heap)[0]
        print(point_heap)


    def fun2():
        n = 7
        point_heap = [(-1, 1), (-2, 1), (-4, 1), (-3, 1)]
        point_heap.extend([(i, 1) for i in b])
        point_heap.sort()
        point_heap = point_heap[:n]
        print(point_heap)


    def fun3():
        n = 7
        point_heap = [(-1, 1), (-2, 1), (-4, 1), (-3, 1)]
        point_heap.extend([(i, 1) for i in b])
        point_heap = sorted(point_heap)[:n]
        print(point_heap)





    def fun100():
        for i in range(1000):
            j = i + 0.0
            k = int(i + 0.0)
            a = j == i
            b = k == i

    profile = line_profiler.LineProfiler(fun100)
    profile.enable()
    fun100()
    profile.disable()
    profile.print_stats()
