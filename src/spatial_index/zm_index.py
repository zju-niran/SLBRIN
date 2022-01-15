import gc
import os

from src.spatial_index.spatial_index import SpatialIndex
from src.b_tree import BTree
from src.spatial_index.common_utils import read_data_and_search
from src.rmi import TrainedNN, AbstractNN


class ZMIndex(SpatialIndex):
    def __init__(self):
        super(ZMIndex, self).__init__("ZM Index")
        self.block_size = 100
        self.total_number = 200000
        self.use_thresholds = [True, False]
        self.thresholds = [5, 1000]
        self.stages = [1, 100]
        self.cores = [[1, 8, 8, 8, 1], [1, 16, 16, 16, 1]]
        self.train_steps = [2000000, 20000]
        self.batch_sizes = [50, 50]
        self.learning_rates = [0.00001, 0.0001]
        self.keep_ratios = [0.9, 1.0]
        self.index = None

    def build(self, points):
        stage_length = len(self.stages)
        train_inputs = [[[] for i in range(self.stages[i])] for i in range(stage_length)]
        train_labels = [[[] for i in range(self.stages[i])] for i in range(stage_length)]
        index = [[None for i in range(self.stages[i])] for i in range(stage_length)]
        train_inputs[0][0] = [point.z for point in points]
        train_labels[0][0] = [point.index for point in points]
        # 构建stage_nums结构的树状NNs
        for i in range(0, stage_length):
            for j in range(0, self.stages[i]):
                if len(train_labels[i][j]) == 0:
                    continue
                inputs = train_inputs[i][j]
                labels = []
                # 非叶子结点决定下一层要用的NN是哪个
                if i < stage_length - 1:
                    # first stage, calculate how many models in next stage
                    divisor = self.stages[i + 1] * 1.0 / (self.total_number / self.block_size)
                    for k in train_labels[i][j]:
                        labels.append(int(k * divisor))
                else:
                    labels = train_labels[i][j]
                # train model
                tmp_index = TrainedNN(inputs, labels,
                                      self.thresholds[i],
                                      self.use_thresholds[i],
                                      self.cores[i],
                                      self.train_steps[i],
                                      self.batch_sizes[i],
                                      self.learning_rates[i],
                                      self.keep_ratios[i])
                tmp_index.train()
                # get parameters in model (weight matrix and bias matrix)
                index[i][j] = AbstractNN(tmp_index.get_weights(),
                                         tmp_index.get_bias(),
                                         self.cores[i],
                                         tmp_index.cal_err())
                del tmp_index
                gc.collect()
                if i < stage_length - 1:
                    # allocate data into training set for models in next stage
                    for ind in range(len(train_inputs[i][j])):
                        # pick model in next stage with output of this model
                        p = index[i][j].predict(train_inputs[i][j][ind])
                        if p > self.stages[i + 1] - 1:
                            p = self.stages[i + 1] - 1
                        train_inputs[i + 1][p].append(train_inputs[i][j][ind])
                        train_labels[i + 1][p].append(train_labels[i][j][ind])

        # 如果叶节点NN的精度低于threshold，则使用Btree来代替
        for i in range(self.stages[stage_length - 1]):
            if index[stage_length - 1][i] is None:
                continue
            mean_abs_err = index[stage_length - 1][i].mean_err
            if mean_abs_err > self.thresholds[stage_length - 1]:
                # replace model with BTree if mean error > threshold
                print("Using BTree")
                index[stage_length - 1][i] = BTree(2)
                index[stage_length - 1][i].build(train_inputs[stage_length - 1][i], train_labels[stage_length - 1][i])
        self.index = index

    def predict(self, point):
        stage_length = len(self.stages)
        pre = 0
        for i in range(0, stage_length):
            pre = self.index[i][pre].predict(point.z)
        return pre


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    path = '../../data/trip_data_2_100000_random_z.csv'
    index = ZMIndex()
    read_data_and_search(path, index, None, None, 7, 0)
