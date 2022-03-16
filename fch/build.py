import os
import numpy
import ray
from torch.utils.data import DataLoader
import torch
import Network as Network
from Dim import Dim

dirname = 'data9-100w'
path_base = "/ArcNas/fch/"


class NetTree:
    def __init__(self, path):
        self.path = path
        self.son = []
        self.divider = None

    def add_son(self, son):
        self.son.append(son)


def init_data(path):
    file = open(path, "r", encoding='UTF-8')
    dim_t = []
    dim_x = []
    dim_y = []
    for line in file:
        cols = line.split(",")
        dim_t.append([float(cols[7]), float(cols[10])])
        dim_x.append([float(cols[8]), float(cols[11])])
        dim_y.append([float(cols[9]), float(cols[12])])
    return dim_t, dim_x, dim_y


def init_min_max(path):
    file = open(path, "r", encoding='UTF-8')
    t_min, t_max, x_min, x_max, y_min, y_max = None, None, None, None, None, None
    for line in file:
        cols = line.split(":")
        if cols[0] == 'T':
            t_min = int(cols[1].split(",")[0])
            t_max = int(cols[1].split(",")[1].replace(' ', ''))
        if cols[0] == 'X':
            x_min = float(cols[1].split(",")[0])
            x_max = float(cols[1].split(",")[1].replace(' ', ''))
        if cols[0] == 'Y':
            y_min = float(cols[1].split(",")[0])
            y_max = float(cols[1].split(",")[1].replace(' ', ''))
    return t_min, t_max, x_min, x_max, y_min, y_max


# 对于复杂的分布 存在简单网络结构无法学习的情况 因此调整网络结构 选取更好的拟合结果
def change_net(net, data, min_val, max_val, min_num, max_num, path):
    hd = 4
    while net.convergence is False:
        hd = hd + 4
        net_temp = Network.auto_train(data, min_val, max_val, min_num, max_num)
        if net.loss > net_temp.loss:
            net = net_temp
        if hd == 12:
            break
    return net


def calc_correlation(la, lb):
    la_m = numpy.sum(la) / (len(la))
    lb_m = numpy.sum(lb) / (len(lb))
    co_s = 0
    for i in range(len(la)):
        for j in range(len(lb)):
            co_s = co_s + (la[i] - la_m) * (lb[j] - lb_m)
    result = co_s / numpy.sqrt(numpy.var(la) * numpy.var(lb))
    return result


def split_data(data, path):
    if len(data) < 100000:
        return [data]
    re = []
    inputs = data[:, 0:1]
    min_i = min(inputs)[0]
    max_i = max(inputs)[0]
    hist = []
    for i in range(1, 11):
        temp = inputs[min_i + (i - 1) * 0.1 * (max_i - min_i) < inputs]
        hist.append(len(temp[temp < min_i + i * 0.1 * (max_i - min_i)]))
    mean_dif = None
    pos = 0
    for i in range(1, 8):
        dif = calc_correlation(hist[0:i + 1], hist[i + 1:10])
        if mean_dif is None or (mean_dif is not None and mean_dif > dif):
            mean_dif = dif
            pos = i + 1
    div = pos * 0.1 * (max_i - min_i) + min_i
    filter_arr = inputs < div
    d1 = []
    d2 = []
    for i in range(len(filter_arr[:, 0])):
        if filter_arr[:, 0][i]:
            d1.append(data[i].tolist())
        else:
            d2.append(data[i].tolist())
    print(len(d1), len(d2))
    d1_arr = split_data(numpy.array(d1), path)
    d2_arr = split_data(numpy.array(d2), path)
    for i in range(len(d1_arr)):
        re.append(d1_arr[i])
    for i in range(len(d2_arr)):
        re.append(d2_arr[i])
    return re


def split_data_average(data):
    # todo 根据输入均分
    inputs = data[:, 0:1]
    min_i = min(inputs)[0]
    max_i = max(inputs)[0]
    slice_num = 10
    range_data = (max_i - min_i) / slice_num
    re = []
    for i in range(1, slice_num + 1):
        temp_data = []
        temp = inputs[inputs < range_data * i + min_i]
        filter_arr = temp > range_data * (i - 1) + min_i
        for j in range(len(filter_arr)):
            if filter_arr[j]:
                temp_data.append(data[j].tolist())
        re.append(temp_data)
    return re


def normalize_data(data, old_min_val, old_max_val, min_val, max_val, old_min_pos, old_max_pos, min_pos, max_pos):
    d = numpy.array(data)
    d[:, 0] = ((d[:, 0] * (old_max_val - old_min_val) + old_min_val) - min_val) / (max_val - min_val)
    d[:, 1] = ((d[:, 1] * (old_max_pos - old_min_pos) + old_min_pos) - min_pos) / (max_pos - min_pos)
    return d.tolist()


# 根据直方图计算分裂
def split_net_hist(net, data, path, file_re, in_min, in_max, out_min, out_max):
    mark = Network.save_model(net, data, path, file_re)
    result = NetTree(net)
    if mark is not True:
        dividers = []
        arr = split_data(numpy.array(data), path)
        pos_start = out_min
        dividers.append(in_min)
        for i in range(len(arr)):
            path_new = path + str(i) + "/"
            if not os.path.exists(path_new):
                os.makedirs(path_new)
            min_val = min(numpy.array(arr[i])[:, 0]) * (in_max - in_min) + in_min
            max_val = max(numpy.array(arr[i])[:, 0]) * (in_max - in_min) + in_min
            dividers.append(max_val)
            min_pos = pos_start
            max_pos = pos_start + len(arr[i]) - 1
            data_sub = normalize_data(arr[i], in_min, in_max, min_val, max_val,
                                      out_min, out_max, min_pos, max_pos)
            temp = Network.auto_train(Dim(data_sub), min_val, max_val, min_pos, max_pos)
            Network.save_model(temp, data_sub, path_new, file_re)
            result.add_son(path_new)
            pos_start = max_pos
        result.divider = dividers
    return result


# todo 直接根据神经网络的预测结果分裂 判断分裂之后是不是数量为0
def split_net_stage(net, data, path, file_re, in_min, in_max, out_min, out_max):
    Network.save_model(net, data, path, file_re)
    dataloader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0, drop_last=False)
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    # 服务器使用cuda
    sub_data_set = [[], [], [], [], [], [], [], [], [], []]
    sub_data_labels = [[], [], [], [], [], [], [], [], [], []]
    num_i = 0
    for item in inputs:
        pre_input = torch.tensor([[item.data.item()]]).cuda()
        pre_output = net.forward(pre_input)
        div = pre_output.cpu().data.numpy()[0][0]
        if div > 1:
            div = 1
        if div < 0:
            div = 0
        pos = min(int(div * 10), 9)
        sub_data_set[pos].append(
            [item.data.item() * (in_max - in_min) + in_min, labels[num_i].data.item() * (out_max - out_min) + out_min])
        num_i = num_i + 1

    for i in range(0, 10):
        path_new = path + str(i) + "/"
        if not os.path.exists(path_new):
            os.makedirs(path_new)
        if sub_data_set[i].__len__() == 0:
            continue
        data = numpy.array(sub_data_set[i])
        min_in = min(data[:, 0])
        max_in = max(data[:, 0])
        data[:, 0] = (data[:, 0] - min_in) / (max_in - min_in)
        min_out = min(data[:, 1])
        max_out = max(data[:, 1])
        data[:, 1] = (data[:, 1] - min_out) / (max_out - min_out)
        net_sub = Network.auto_train(Dim(data.tolist()), min_in, max_in, min_out, max_out, path_new)
        Network.save_model(net_sub, data.tolist(), path_new, file_re)
    return 1


def save_split(net, data, path, file_re, in_min, in_max, out_min, out_max):
    path1 = path + "hist/"
    path2 = path + "stage/"
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    Network.save_model(net, data, path, file_re)
    # tree1 = split_net_hist(net, data, path1, file_re, in_min, in_max, out_min, out_max)
    split_net_stage(net, data, path2, file_re, in_min, in_max, out_min, out_max)
    # f1 = open(path1 + 'tree_hist', 'wb')

    # pickle.dump(tree1, f1)
    # pickle.dump(tree2, f2)
    # f1.close()


def train_net(result_path):
    file_re = open(result_path, "w", encoding='UTF-8')
    for i in range(0, 13):
        data_t, data_x, data_y = init_data(
            path_base + dirname + "/" + str(i) + "/part-00000")
        t_min, t_max, x_min, x_max, y_min, y_max = init_min_max(
            path_base + dirname + "/" + str(i) + "-info.meta")

        # path_init_pic = path_base+"graphs-"+dirname+"/"
        # split_data(numpy.array(data_t), path_init_pic + str(i) + "-t")
        # split_data(numpy.array(data_x), path_init_pic + str(i) + "-x")
        # split_data(numpy.array(data_y), path_init_pic + str(i) + "-y")
        # td = Dim(data_t)
        # xd = Dim(data_x)
        # yd = Dim(data_y)
        #
        # # # 保存初始图像
        #
        # Network.save_graph_show(path_init_pic, str(i) + "-t", td, batch_size=len(td))
        # Network.save_graph_show(path_init_pic, str(i) + "-x", xd, batch_size=len(xd))
        # Network.save_graph_show(path_init_pic, str(i) + "-y", yd, batch_size=len(yd))
        # dataloader = DataLoader(Dim(data_t), batch_size=len(data_t), shuffle=False, num_workers=0, drop_last=False)
        # data_iter = iter(dataloader)
        # inputs, labels = next(data_iter)
        # # 服务器使用cuda
        # for item in inputs:
        #     print(item)
        folder = path_base + "graphs-" + dirname + "/train-result/" + str(i)

        path_x = folder + '-x/'
        path_y = folder + '-y/'
        path_t = folder + '-t/'
        if not os.path.exists(path_x):
            os.makedirs(path_x)
        if not os.path.exists(path_y):
            os.makedirs(path_y)
        if not os.path.exists(path_t):
            os.makedirs(path_t)
        net_x = Network.auto_train(Dim(data_x), x_min, x_max, 0, len(data_x) - 1, path_x)
        net_y = Network.auto_train(Dim(data_y), y_min, y_max, 0, len(data_y) - 1, path_y)
        net_t = Network.auto_train(Dim(data_t), t_min, t_max, 0, len(data_t) - 1, path_t)
        save_split(net_x, data_x, path_x, file_re, x_min, x_max, 0, len(data_x) - 1)
        save_split(net_y, data_y, path_y, file_re, y_min, y_max, 0, len(data_y) - 1)
        save_split(net_t, data_t, path_t, file_re, t_min, t_max, 0, len(data_t) - 1)
    file_re.close()

    # Press the green button in the gutter to run the script.


if __name__ == '__main__':
    # print("父进程：{}".format(os.getpid()))
    # pool = mp.Pool(processes=20)
    # for i in range(20):
    #     pool.apply_async(train_net, args=[path_base + "graphs-" + dirname + "/train-result/errs.txt"])
    # pool.close()
    # pool.join()
    ray.init()
    train_net(path_base + "graphs-" + dirname + "/train-result/errs.txt")

# f1 = open("C:/Users/ph/Desktop/testData/graphs-data6-1w/train-result/0-t/net_tree", 'rb')
# treenet = pickle.load(f1)
# print(treenet)
# Network.load_model(path_t)
# torch.save(net_t.state_dict(), path_t + "-weights")

# net_t = Network.train(td, 1, 1, 'threshold', batch_size=10000)
# net_x = Network.train(xd, 1, 1, 'threshold', 'C:/Users/ph/Desktop/testData/graphs-data6/train-result/62-x',
#                       batch_size=1000)
# for i in range(4, 32):
#     net = Network.train(xd, 1, 1, 'threshold',
#                         'C:/Users/ph/Desktop/testData/graphs-data6/train-result/0-x/' + str(i) + '-0-x',
#                         hidden_dims=i,
#                         batch_size=1000)
#     if net.convergence:
#         break
