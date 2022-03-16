import line_profiler
import pandas as pd

from src.spatial_index.common_utils import Region

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
    def fun2(a):
        if a[0][0] == 0:  # no relation
            return 1
        elif a[0][0] == 2:  # block contain window
            return 2
        elif a[0][0] == 1:
            return 3
        else:
            return 4


    def fun3(a):
        b = a[0][0]
        if b == 0:  # no relation
            return 1
        elif b == 2:  # block contain window
            return 2
        elif b == 1:
            return 3
        else:
            return 4


    def fun1():
        for i in range(1000):
            a = (
            (1, Region(0, 0, 1, 1), 0, [0, 1]), (2, Region(0, 0, 1, 1), 0, [0, 1]), (3, Region(0, 0, 1, 1), 0, [0, 1]),
            (4, Region(0, 0, 1, 1), 0, [0, 1]))
            for a1 in a:
                b = fun2(a)
                c = fun3(a)


    index_list = pd.read_csv('D:\Code\Paper\st-learned-index\src\spatial_index\model\gm_index_10w\index_list.csv',
                             float_precision='round_trip')
    profile = line_profiler.LineProfiler(fun1)
    profile.enable()
    fun1()
    profile.disable()
    profile.print_stats()
