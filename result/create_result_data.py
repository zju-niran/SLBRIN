import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams, cm, dates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from src.experiment.common_utils import Distribution, data_region, build_data_path

config = {
    "font.size": 20,
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['FangSong'],
}
rcParams.update(config)


def plot_scatter(x, y, model_path, x_title, y_title, lim, adjust):
    plt.figure(figsize=(6.4, 6.144))
    scatter_size = 8
    plt.scatter(x, y, s=scatter_size)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.gcf().subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3])
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_date_lines(x, y_list, model_path, x_title, y_title, colors, ylim, adjust):
    plt.figure(figsize=(6.4, 6.144))
    for i in range(len(y_list)):
        plt.plot(x, y_list[i], color=colors[i])
    plt.gca().xaxis.set_major_locator(dates.DayLocator(interval=14))
    plt.gca().xaxis.set_minor_locator(dates.DayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%m-%d"))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    if ylim:
        plt.gca().yaxis.set_major_locator(MultipleLocator(ylim[2]))
        plt.ylim(ylim[0], ylim[1])
    # plt.gcf().autofmt_xdate()
    plt.gcf().subplots_adjust(left=adjust[0], right=adjust[1], bottom=adjust[2], top=adjust[3])
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_data_o(output_path):
    data_distributions = [Distribution.UNIFORM_10W, Distribution.NORMAL_10W, Distribution.NYCT_10W]
    for data_distribution in data_distributions:
        data_list = np.load(build_data_path[data_distribution][3:], allow_pickle=True)
        sample_key = np.random.randint(0, len(data_list), size=10000)
        data_list = data_list[sample_key]
        region = data_region[data_distribution]
        x = [(data[0] - region.left) / (region.right - region.left) for data in data_list]
        y = [(data[1] - region.bottom) / (region.up - region.bottom) for data in data_list]
        plot_scatter(x, y, output_path + "/data1_%s.png" % data_distribution.name,
                     '${x}$', '${y}$', [0, 1, 0, 1], [0.13, 0.97, 0.12, 0.98])

def plot_data_F_n(input_path, output_path):
    """
    统计每小时的空间分布，并出图
    """
    start_time = 1356998400
    end_time = 1362096000  # 1359676800
    time_interval = 60 * 60
    cdf_width = 10
    # cdf_width = 100
    time_id = math.ceil((end_time - start_time) / time_interval)
    time_id_list = [datetime.fromtimestamp(d).date() for d in range(start_time, end_time, time_interval)]
    colors = cm.rainbow(np.arange(cdf_width) / cdf_width)
    data_distributions = [Distribution.UNIFORM, Distribution.NORMAL, Distribution.NYCT]
    for data_distribution in data_distributions:
        # data_list1 = np.load(build_data_path[data_distribution][3:], allow_pickle=True)
        # data_list2 = np.load(update_data_path[data_distribution][3:], allow_pickle=True)
        # geohash = Geohash.init_by_precision(data_precision=data_precision[data_distribution],
        #                                     region=data_region[data_distribution])
        # min_key = 0
        # max_key = 1 << geohash.sum_bits
        # key_interval = (max_key - min_key) / cdf_width
        # key_list = [int(min_key + k * key_interval) for k in range(cdf_width)]
        # old_cdfs = [[] for k in range(time_id)]
        # for data in data_list1:
        #     old_cdfs[int((data[2] - start_time) // time_interval)].append(geohash.encode(data[0], data[1]))
        # for data in data_list2:
        #     old_cdfs[int((data[2] - start_time) // time_interval)].append(geohash.encode(data[0], data[1]))
        # old_max_keys = [len(cdf) / 10000 for cdf in old_cdfs]
        # for k in range(len(old_cdfs)):
        #     old_cdfs[k].sort()
        #     x_len = len(old_cdfs[k])
        #     x_max_key = x_len - 1
        #     tmp = []
        #     p = 0
        #     for l in range(cdf_width):
        #         p = binary_search_less_max_duplicate(old_cdfs[k], key_list[l], p, x_max_key)
        #         tmp.append(p / x_len)
        #     old_cdfs[k] = tmp
        data_F_path = os.path.join(input_path, 'data_F_%s_%s.csv' % (cdf_width, data_distribution.name))
        data_n_path = os.path.join(input_path, 'data_n_%s_%s.csv' % (cdf_width, data_distribution.name))
        # data_F = pd.DataFrame(list(map(list, zip(*old_cdfs))))
        # data_F.to_csv(data_F_path, header=False, index=False)
        # data_n = pd.DataFrame([old_max_keys])
        # data_n.to_csv(data_n_path, header=False, index=False)
        data_F = pd.read_csv(data_F_path, header=None).values
        plot_date_lines(time_id_list, data_F,
                        output_path + "/data2_%s_%s.png" % (cdf_width, data_distribution.name),
                        '${t}$（${h}$）', '${F}$', colors, None, [0.13, 0.97, 0.12, 0.98])
        data_n = pd.read_csv(data_n_path, header=None).values
        plot_date_lines(time_id_list, data_n, output_path + "/data3_%s.png" % data_distribution.name,
                        '${t}$（${h}$）', '${n}$（$\mathrm{×10000}$）', colors, [0.0, 4.0, 1.0], [0.13, 0.97, 0.12, 0.98])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    output_path = "./result_data"
    plot_data_o(output_path)
    plot_data_F_n(output_path, output_path)
