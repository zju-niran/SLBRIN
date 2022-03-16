import time
from functools import wraps

import matplotlib.pyplot as plt
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.tune.logger import LoggerCallback
from ray.tune.schedulers import HyperBandScheduler
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)


class TestLoggerCallback(LoggerCallback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"TestLogger for trial {trial}: {result}")


class Net(nn.Module):
    def __init__(self, input_dims, hidden_dims_1, hidden_dims_2, hidden_dims_3, output_dims, input_min, input_max,
                 output_min,
                 output_max):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(input_dims, hidden_dims_1)  # 隐藏层的线性输出
        self.hidden2 = torch.nn.Linear(hidden_dims_1, hidden_dims_2)  # 隐藏层的线性输出
        self.hidden3 = torch.nn.Linear(hidden_dims_2, hidden_dims_3)  # 隐藏层的线性输出
        self.result = torch.nn.Linear(hidden_dims_3, output_dims)  # 输出层的线性输出
        self.convergence = False
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max
        self.loss = None
        self.divider = None

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = (self.result(x))
        return x

    # todo 改成直接运算 乘以权重

    def my_predict(self, x):
        x = (x - self.input_min) / (self.input_max - self.input_min)
        x = self.forward(x)
        x[x < 0] = 0
        x[x > 1] = 1
        x = x * (self.output_max - self.output_min) + self.output_min
        return x.int()


def draw_graph(x, y, pre, color, hidden_dims, times, loss, save=False, path=""):
    plt.ioff()
    plt.scatter(x.cpu().data.numpy(), y.cpu().data.numpy())
    plt.scatter(x.cpu().data.numpy(), pre.cpu().data.numpy(), edgecolors=color)
    plt.text(0.8, 1, "step:" + str(times))
    plt.text(0.8, 0.9, "loss:" + str(loss))
    plt.text(0.8, 0.8, "hidden_dims:" + str(hidden_dims))
    if save:
        plt.savefig(path)
    # plt.show()
    plt.cla()


def save_graph_show(path, filename, data, batch_size=1000):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    plt.ioff()
    plt.scatter(inputs.data.numpy(), labels.data.numpy())
    plt.savefig(path + filename)
    # plt.show()
    plt.cla()
    plt.ioff()
    plt.hist(inputs.data.numpy(), 10)
    plt.savefig(path + filename + "-10-hist")
    # plt.show()
    plt.cla()


def save_model(net, data, path, re_path):
    torch.save(net.state_dict(), path + "weights")
    file_err = open(path + 'err_bounds', "w", encoding='UTF-8')
    file_mm = open(path + 'min_max', "w", encoding='UTF-8')
    min_b, max_b, presc = calc_max_means(net, data)
    cols = path.split("/")
    node = cols[len(cols) - 2]
    file_re = re_path
    file_re.write(node + ";" + str(min_b) + ";" + str(max_b) + "\n")
    file_err.write(str(net.loss) + "," + str(len(data)) + "," +
                   str(min_b) + "," + str(max_b))
    file_mm.write(str(net.input_min) + "," + str(net.input_max) + "," +
                  str(net.output_min) + "," + str(net.output_max))
    file_err.close()
    file_mm.close()
    if min_b + max_b > 500 and len(data) > 100000:
        return presc
    return True


# 加载模型 并且读取获得最大最小误差
def load_model(path):
    states = torch.load(path + "weights")
    file_err = open(path + 'err_bounds', "r", encoding='UTF-8')
    file_mm = open(path + 'min_max', "r", encoding='UTF-8')
    min_b, max_b = None, None
    min_in, max_in, min_out, max_out = None, None, None, None
    for line in file_err:
        cols = line.split(",")
        min_b = float(cols[2])
        max_b = float(cols[3])
    for line in file_mm:
        cols = line.split(",")
        min_in = float(cols[0])
        max_in = float(cols[1])
        min_out = float(cols[2])
        max_out = float(cols[3])
    hidden = states['hidden1.weight'].shape[0]
    net = Net(1, hidden, 1, min_in, max_in, min_out, max_out)
    net.load_state_dict(states)
    net.eval()
    return net, min_b, max_b


# 重复序列二分搜索
def binary_search_r_min(arr, left, right, key):
    mid = int((left + right) / 2)
    result = None
    while left < right:
        if arr[mid] == key:
            result = mid
            return result
        if arr[mid] < key:
            left = mid
        else:
            right = mid
        mid = int((left + right) / 2)
        if left == mid:
            if arr[left] == key:
                result = left
                return result
            elif arr[right] == key:
                result = right
                return result
            elif arr[left] < key < arr[right]:
                result = right
                return result
            break
    return result


def binary_search_r_max(arr, left, right, key):
    mid = int((left + right) / 2)
    result = None
    while left < right:
        if arr[mid] == key:
            result = mid
            return result
        if arr[mid] <= key:
            left = mid
        else:
            right = mid
        mid = int((left + right) / 2)
        if left == mid:
            if arr[right] == key:
                result = right
                return result
            elif arr[left] == key:
                result = left
                return result
            elif arr[left] < key < arr[right]:
                result = left
                return result
            break
    return result


def range_index(net, arr, min_b, max_b, key_min, key_max, min_num, max_num):
    r_min = []
    r_max = []
    all_key = None
    input_key_min = torch.from_numpy(key_min)
    input_key_max = torch.from_numpy(key_max)
    if min_num == 0 and max_num == 0:
        return r_min, r_max
    elif min_num == 0:
        all_key = input_key_max
    elif max_num == 0:
        all_key = input_key_min
    else:
        all_key = torch.cat([input_key_min, input_key_max], dim=0)

    start = time.perf_counter_ns()
    pres = net.my_predict(all_key)
    end = time.perf_counter_ns()
    print(end - start)

    start = time.time()
    for i in range(min_num):
        pre = None
        if key_min[i][0] < arr[net.output_min]:
            pre = net.output_min
        else:
            pm = pres[i][0].item()
            if arr[pm] > key_min[i][0]:
                pre = binary_search_r_min(arr, max(pm - min_b, net.output_min), pm, key_min[i][0])
            elif arr[pm] < key_min[i][0]:
                pre = binary_search_r_min(arr, pm, min(pm + max_b, net.output_max), key_min[i][0])
        r_min.append(pre)
    for i in range(max_num):
        pre = None
        if key_max[i][0] > arr[net.output_max]:
            pre = net.output_max
        else:
            pm = pres[i + min_num][0].item()
            if arr[pm] > key_max[i][0]:
                pre = binary_search_r_max(arr, max(pm - min_b, net.output_min), pm, key_max[i][0])
            elif arr[pm] < key_max[i][0]:
                pre = binary_search_r_max(arr, pm, min(pm + max_b, net.output_max), key_max[i][0])
        r_max.append(pre)
    end = time.time()
    print(start, end)
    # 前者是大于key的第一个 后者是小于key的最后一个
    return r_min, r_max


def calc_max_means(net, data):
    dataloader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0, drop_last=False)
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    # 服务器使用cuda
    inputs_batch = torch.tensor([inputs.data.numpy()]).T.cuda()
    labels_batch = torch.tensor([labels.data.numpy()]).T.cuda()
    pres = net.cuda().forward(inputs_batch)
    presc = pres
    presc[presc > 1] = 1
    presc[presc < 0] = 0
    max_mean = len(data) * max(presc - labels_batch)
    min_mean = len(data) * abs(min(presc - labels_batch))
    return int(min_mean.cpu().data.numpy()[0]) + 1, int(max_mean.cpu().data.numpy()[0]) + 1, presc.cpu()


def train4autoML(config, checkpoint_dir=None):
    data, input_min, input_max, output_min, output_max = config["data"], config["input_min"], config["input_max"], \
                                                         config["output_min"], config["output_max"]
    data = ray.get(data)
    hidden_dims_1, hidden_dims_2, hidden_dims_3, lr, batch_size = config["h1"], config["h2"], config["h3"], config[
        "lr"], config["batch_size"]

    net = Net(1, hidden_dims_1, hidden_dims_2, hidden_dims_3, 1, input_min, input_max, output_min, output_max).cuda()
    net.train()
    # 定义损失函数
    criterion = nn.MSELoss()
    batch_size = min(batch_size, int(len(data) / 10))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    data_iter = iter(dataloader)
    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # 自动修改学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    all_loss = 0
    # 输入数据，输出结果
    # int(len(data) / batch_size)
    for i in range(int(len(data) / batch_size)):
        all_loss = 0
        inputs, labels = next(data_iter)
        inputs_batch = torch.tensor([inputs.data.numpy()]).T.cuda()
        labels_batch = torch.tensor([labels.data.numpy()]).T.cuda()
        # print('epoch: %d' % i)
        j = 0
        last_error = None
        while True:
            # 梯度清零
            optimizer.zero_grad()
            outputs = net(inputs_batch)
            criterion.cuda()
            loss = criterion(outputs, labels_batch)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 每200步长 迭代检查误差
            if j % 1000 == 0:
                # print('[%d] loss: %.12f learningRate: %.9f' % (
                #    j + 1, loss.data.item(), optimizer.state_dict()['param_groups'][0]['lr']))
                scheduler.step(loss.data.item())

            # 当误差达到阈值或者最大迭代次数停止训练 将最大误差存下来
            if (last_error is not None and loss.data.item() >= last_error) or j >= 100000 or loss.data.item() < 0.0001:
                if loss.data.item() < 0.0001:
                    net.convergence = True
                net.loss = all_loss / (j + 1)
                # draw_graph(inputs_batch, labels_batch, outputs, "red", hidden_dims, j, loss.data.item(), True,
                #            path + str(hidden_dims) + "-" + str(i))
                break
            all_loss += loss.data.item()
            last_error = loss.data.item()
            j += 1
        tune.report(loss=net.loss)


def train(config, data, input_min, input_max, output_min, output_max):
    # 重复训练10次 取最好的结果
    last_net_loss = None
    result_net = None
    times = 3
    hidden_dims_1, hidden_dims_2, hidden_dims_3, lr, batch_size, path = config["h1"], config["h2"], config["h3"], \
                                                                        config[
                                                                            "lr"], config["batch_size"], config["path"]

    while times > 0:
        net = Net(1, hidden_dims_1, hidden_dims_2, hidden_dims_3, 1, input_min, input_max, output_min,
                  output_max).cuda()
        net.train()
        # 定义损失函数
        criterion = nn.MSELoss()
        batch_size = min(batch_size, int(len(data) / 10))
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        data_iter = iter(dataloader)
        # 定义优化器
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        # 自动修改学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        all_loss = 0
        # 输入数据，输出结果
        # int(len(data) / batch_size)
        for i in range(int(len(data) / batch_size)):
            print("epoch:" + str(i))
            inputs, labels = next(data_iter)
            inputs_batch = torch.tensor([inputs.data.numpy()]).T.cuda()
            labels_batch = torch.tensor([labels.data.numpy()]).T.cuda()
            # print('epoch: %d' % i)
            j = 0
            last_error = None
            while True:
                # 梯度清零
                optimizer.zero_grad()
                outputs = net(inputs_batch)
                criterion.cuda()
                loss = criterion(outputs, labels_batch)
                loss.backward()

                # 更新参数
                optimizer.step()

                # 每200步长 迭代检查误差
                if j % 1000 == 0:
                    # print('[%d] loss: %.12f learningRate: %.9f' % (
                    #    j + 1, loss.data.item(), optimizer.state_dict()['param_groups'][0]['lr']))
                    scheduler.step(loss.data.item())

                # 当误差达到阈值或者最大迭代次数停止训练 将最大误差存下来
                if (
                        last_error is not None and loss.data.item() >= last_error) or j >= 100000 or loss.data.item() < 0.0001:
                    if loss.data.item() < 0.0001:
                        net.convergence = True
                    all_loss += loss.data.item()
                    draw_graph(inputs_batch, labels_batch, outputs, "red",
                               str(hidden_dims_1) + "-" + str(hidden_dims_2) + "-" + str(hidden_dims_3)+"-"+str(times), j,
                               loss.data.item(), True,
                               path + str(times)+"-"+str(hidden_dims_1) + "-" + str(hidden_dims_2) + "-" + str(
                                   hidden_dims_3) + "-" + str(i))
                    break

                last_error = loss.data.item()
                j += 1
        net.loss = all_loss / int(len(data) / batch_size)
        if last_net_loss is None:
            result_net = net
            last_net_loss = net.loss
        elif net.loss < last_net_loss:
            result_net = net
            last_net_loss = net.loss
        if last_net_loss < 0.0003:
            break
        times = times - 1
    print(hidden_dims_1, hidden_dims_2, lr, batch_size)
    print(result_net.loss)
    return result_net


def auto_train(data, imin, imax, omin, omax, path):
    config_fch = {
        "h1": 8,
        "h2": 8,
        "h3": 8,
        "lr": 0.0001,
        "batch_size": 100000,
        "input_min": imin,
        "input_max": imax,
        "output_min": omin,
        "output_max": omax,
        "path": path
    }
    # print("define config")
    # hyperband = HyperBandScheduler(metric="loss", mode="min")
    #
    # analysis = tune.run(
    #     train4autoML,
    #     metric="loss", mode="min",
    #     config=config_fch,
    #     local_dir="/ArcNas/fch/ray-log",
    #     log_to_file=True,
    #     max_concurrent_trials=9,
    #     callbacks=[TestLoggerCallback()],
    #     # log_to_file=True,
    #     resources_per_trial={'cpu': 10, 'gpu': 1})
    #
    # print("success, get the best config.")
    #
    # best_config = analysis.get_best_config(metric="loss", mode="min")
    # best_config = analysis.get_best_config(metric="loss", mode="min")

    print("get_best_config:")
    print(config_fch)
    net = train(config_fch, data, imin, imax, omin, omax)
    return net

# def moral():
#     return
