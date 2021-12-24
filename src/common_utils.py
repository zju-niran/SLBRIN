import time

import pandas as pd


class Point:
    def __init__(self, lng, lat, index=0):
        self.lng = lng
        self.lat = lat
        self.index = index

    def __eq__(self, other):
        if other.lng == self.lng and other.lat == self.lat:
            return True
        else:
            return False

    def __str__(self):
        return "Point({0}, {1}, {2})".format(self.lng, self.lat, self.index)

    def near(self, other):
        """
        近似相等，只要整数部分一致即可
        :param other:
        :return:
        """
        if int(other.lng) == int(self.lng) and int(other.lat) == int(self.lat):
            return True
        else:
            return False


def read_data_and_search(path, index):
    index_name = index.name
    data = pd.read_csv(path, header=None)
    train_set_point = []
    test_set_point = []
    test_ratio = 0.5  # 测试集占总数据集的比例
    for i in range(int(data.shape[0])):
        train_set_point.append(Point(data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2]))
    test_set_point = train_set_point[:int(len(train_set_point) * test_ratio)]

    print("*************start %s************" % index_name)
    print("Start Build")
    start_time = time.time()
    index.build(train_set_point)
    end_time = time.time()
    build_time = end_time - start_time
    print("Build %s time " % index_name, build_time)
    err = 0
    print("Calculate error")
    start_time = time.time()
    for ind in range(len(test_set_point)):
        pre = index.predict(test_set_point[ind])
        err += abs(pre - test_set_point[ind].index)
        if err != 0:
            flag = 1
            pos = pre
            off = 1
            while pos != test_set_point[ind].index:
                pos += flag * off
                flag = -flag
                off += 1
    end_time = time.time()
    search_time = (end_time - start_time) / len(test_set_point)
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_point)
    print("mean error = ", mean_error)
    print("*************end %s************" % index_name)
