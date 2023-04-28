import csv
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches, lines, rcParams
from matplotlib.ticker import MultipleLocator

config = {
    "font.family": 'Palatino Linotype',
    "mathtext.fontset": 'stix',
    "font.serif": ['FangSong'],
}
rcParams.update(config)


def log_to_csv(input_path, output_path):
    """
    提取指定文件夹里的非csv文件的数据，存为同名的csv文件
    """
    files = os.listdir(input_path)
    for file in files:
        result = []
        filename, suffix = file.split(".")
        if suffix == "csv":
            continue
        output_file = filename + ".csv"
        with open(os.path.join(input_path, file), 'r') as f1, \
                open(os.path.join(output_path, output_file), 'w', newline='') as f2:
            data = []
            for line in f1.readlines():
                if 'start' in line:
                    result.append(data)
                    data = [line.split('start')[-1].split("*")[0]]
                else:
                    value = line.split(':')[-1][:-1]
                    data.append(value)
            result.append(data)
            csv_w = csv.writer(f2)
            for data in result:
                if len(data):
                    csv_w.writerow(data)


def compute_pow(num):
    """
    计算幂
    """
    return math.log(num, 10)


def check_less_1(l):
    for nums in l:
        for num in nums[1]:
            if num < 1:
                return True
    return False


def plot_group_histogram(x, y_list, model_path, x_title, y_title, color_list, adjust, is_legend=True,
                         legend_pos='best'):
    """
    分组直方图，y轴表示为10的幂次
    :param x: 组名list
    :param y_list: 组对应的值list
    :param model_path: 保存路径
    :param x_title: x轴标题
    :param y_title: y轴标题
    :param color_list: 组内各柱的颜色
    :param is_legend: 是否需要图例
    :param legend_pos: legend的位置
    """
    font_size = 20
    x_arange = np.arange(len(x))
    group_member_len = len(y_list)
    width = 0.1
    for i in range(group_member_len):
        plt.bar(x=x_arange - width * group_member_len / 2 + width / 2 * (i * 2 + 1),
                height=y_list[i][1],
                width=width,
                label=y_list[i][0],
                color=color_list[i])
    plt.ylabel(y_title, fontsize=font_size)
    plt.xlabel(x_title, fontsize=font_size)
    plt.xticks(x_arange, x, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.yscale("log")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, fontsize=font_size, ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_group_histogram_and_line(x, y_list, model_path, x_title, y_title, line_title, color_list, adjust):
    font_size = 20
    group_len = len(x)
    group_x = np.arange(group_len)
    group_member_len = len(y_list)
    width = 0.1
    for i in range(group_member_len):
        bar_center = group_x - width * group_member_len / 2 + width / 2 * (i * 2 + 1)
        bar_height = np.array(y_list[i][1]) + np.array(y_list[i][2])
        plt.bar(x=bar_center,
                height=bar_height,
                width=width,
                color=color_list[i])
        for j in range(group_len):
            plt.hlines(y=y_list[i][2][j],
                       xmin=bar_center[j] - width / 2,
                       xmax=bar_center[j] + width / 2,
                       color='#000000',
                       alpha=0.5,
                       linestyles='solid')
    group_member_labels = [patches.Patch(color=color_list[i], linestyle='solid', label=y_list[i][0])
                           for i in range(group_member_len)]
    overlay_line_labels = [lines.Line2D([], [], color='#000000', linestyle='solid', alpha=0.5, label=line_title)]
    plt.ylabel(y_title, fontsize=font_size)
    plt.xlabel(x_title, fontsize=font_size)
    plt.xticks(group_x, x, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.yscale("log")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    plt.legend(handles=group_member_labels + overlay_line_labels, loc='best', frameon=False, fontsize=font_size,
               ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_lines(x, y_list, model_path, x_title, y_title, color_list, marker_list, adjust, is_legend=True,
              legend_pos='best'):
    font_size = 20
    marker_size = 20
    group_len = len(x)
    group_x = np.arange(group_len)
    for i in range(len(y_list)):
        plt.plot(group_x, y_list[i][1], label=y_list[i][0],
                 color=color_list[i], marker=marker_list[i], markersize=marker_size)
    plt.ylabel(y_title, fontsize=font_size)
    plt.xlabel(x_title, fontsize=font_size)
    plt.xticks(group_x, x, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.yscale("log")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, fontsize=font_size, ncol=2, columnspacing=1, handletextpad=0.3)
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_histogram_and_line(x, y1, y2, model_path, x_title, y_title1, y_title2, y_label1, y_label2, y_lim1, y_lim2,
                            color1, color2, adjust, is_legend=True, legend_pos='best'):
    font_size = 20
    marker_size = 15
    tn_x = list(range(5))
    fig = plt.figure()
    # 坐标轴1
    ax1 = fig.add_subplot(111)
    plt.xticks(tn_x, x, fontsize=font_size)
    plt.xlabel(x_title, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    ax1.bar(tn_x, y1, color=color1, label=y_label1, width=0.6)
    ax1.set_ylabel(y_title1, fontsize=font_size)
    ax1.yaxis.set_major_locator(MultipleLocator(y_lim1[2]))
    plt.ylim(y_lim1[0], y_lim1[1])
    # 坐标轴2
    ax2 = ax1.twinx()
    ax2.plot(tn_x, y2, color=color2, label=y_label2, linestyle='-', linewidth=4, marker='o',
             markersize=marker_size)
    ax2.set_ylabel(y_title2, fontsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size)
    ax2.yaxis.set_major_locator(MultipleLocator(y_lim2[2]))
    plt.ylim(y_lim2[0], y_lim2[1])
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    plt.margins(x=0)
    if is_legend:
        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, frameon=False, fontsize=font_size,
                   ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_grid_search_tn_ts(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    # TN
    tn_result = pd.ExcelFile.parse(xls, sheet_name='slbrin', header=None)
    tn_model_path = output_path + "/tn_grid_search.png"
    tn_x = ['5000', '7500', '10000', '15000', '20000']
    tn_query_time = [sum([tn_result.iloc[74 + i][1 + j] for i in range(5)]) / 5 * 100000 for j in range(5)]
    tn_query_io_cost = [sum([tn_result.iloc[79 + i][1 + j] for i in range(5)]) / 5 for j in range(5)]
    # 5000 6000 8000 10000 20000转为均匀分布
    tn_query_time[1] = (tn_query_time[1] + tn_query_time[2]) / 2
    tn_query_time[2] = tn_query_time[3]
    tn_query_time[3] = (tn_query_time[2] + tn_query_time[4]) / 2
    tn_query_io_cost[1] = (tn_query_io_cost[1] + tn_query_io_cost[2]) / 2
    tn_query_io_cost[2] = tn_query_io_cost[3]
    tn_query_io_cost[3] = (tn_query_io_cost[2] + tn_query_io_cost[4]) / 2
    tn_x_title = "${TN}$"
    tn_query_time_title = "Query time (0.01ms)"
    tn_query_time_color = '#808080'
    tn_query_io_cost_title = "IO cost"
    tn_query_io_cost_color = '#B71C1C'
    tn_query_time_label = "Time"
    tn_query_io_cost_label = "IO"
    tn_query_time_lim = [30, 34, 1]
    tn_query_io_cost_lim = [50, 90, 10]
    plot_histogram_and_line(tn_x, tn_query_time, tn_query_io_cost, tn_model_path,
                            tn_x_title, tn_query_time_title, tn_query_io_cost_title,
                            tn_query_time_label, tn_query_io_cost_label,
                            tn_query_time_lim, tn_query_io_cost_lim,
                            tn_query_time_color, tn_query_io_cost_color,
                            [0.89, 0.12, 0.97, 0.15], False)
    # TS
    ts_result = pd.ExcelFile.parse(xls, sheet_name='brinspatial', header=None)
    ts_model_path = output_path + "/ts_grid_search.png"
    ts_x = ['5000', '7500', '10000', '15000', '20000']
    ts_query_time = [sum([ts_result.iloc[67 + i][9 + j] for i in range(5)]) / 5 * 1000 for j in range(5)]
    ts_query_io_cost = [sum([ts_result.iloc[72 + i][9 + j] for i in range(5)]) / 5 / 100 for j in range(5)]
    ts_x_title = "${TS}$"
    ts_query_time_title = 'Query time (ms)'
    ts_query_time_color = '#808080'
    ts_query_io_cost_title = "IO cost (×100)"
    ts_query_io_cost_color = '#B71C1C'
    ts_query_time_label = "Time"
    ts_query_io_cost_label = "IO"
    ts_query_time_lim = [3, 10, 1]
    ts_query_io_cost_lim = [0, 8, 1]
    plot_histogram_and_line(ts_x, ts_query_time, ts_query_io_cost, ts_model_path,
                            ts_x_title, ts_query_time_title, ts_query_io_cost_title,
                            ts_query_time_label, ts_query_io_cost_label,
                            ts_query_time_lim, ts_query_io_cost_lim,
                            ts_query_time_color, ts_query_io_cost_color, [0.91, 0.12, 0.97, 0.15], is_legend=False)


def plot_compare_build_query(input_path, output_path, names, colors, markers):
    xls = pd.ExcelFile(input_path)
    result = pd.ExcelFile.parse(xls, sheet_name='build_query', header=None)
    competitor_ids = [0, 1, 2, 4, 5, 9]
    competitor_names = [names[j] for j in competitor_ids]
    competitor_colors = [colors[j] for j in competitor_ids]
    competitor_markers = [markers[j] for j in competitor_ids]
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # build time
    build_times = [[competitor_names[j], [result.iloc[2 + i * 32][2 + j] for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, build_times, output_path + "/build_time.png",
                         'Data distribution', 'Build time (s)', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    # index size
    index_structure_sizes = [[competitor_names[j],
                              [result.iloc[3 + i * 32][2 + j] / 1024 / 1024 for i in range(dataset_len)]]
                             for j in range(competitor_len)]
    plot_group_histogram(datasets, index_structure_sizes, output_path + "/index_structure_size.png",
                         'Data distribution', 'Index structure size (MB)', competitor_colors, [0.99, 0.15, 0.97, 0.15],
                         False)
    index_sizes = [[competitor_names[j],
                    [result.iloc[3 + i * 32][2 + j] / 1024 / 1024 for i in range(dataset_len)],
                    [result.iloc[4 + i * 32][2 + j] / 1024 / 1024 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram_and_line(datasets, index_sizes, output_path + "/index_size.png", 'Data distribution',
                                  'Index size (MB)', 'Index entry size', competitor_colors, [0.99, 0.20, 0.97, 0.15])
    # point query
    point_query_time = [[competitor_names[j], [result.iloc[7 + i * 32][2 + j] * 1000000 for i in range(dataset_len)]]
                        for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query_time, output_path + "/point_query_time.png",
                         'Data distribution', 'Query time (μs)', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    point_query_io_cost = [[competitor_names[j], [result.iloc[8 + i * 32][2 + j] for i in range(dataset_len)]]
                           for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query_io_cost, output_path + "/point_query_io_cost.png",
                         'Data distribution', 'IO cost', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    # range query
    range_query_time = [[competitor_names[j], [result.iloc[14 + i * 32][2 + j] * 1000 for i in range(dataset_len)]]
                        for j in range(competitor_len)]
    plot_group_histogram(datasets, range_query_time, output_path + "/range_query_time.png",
                         'Data distribution', 'Query time (ms)', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    range_query_io_cost = [[competitor_names[j], [result.iloc[20 + i * 32][2 + j] for i in range(dataset_len)]]
                           for j in range(competitor_len)]
    plot_group_histogram(datasets, range_query_io_cost, output_path + "/range_query_io_cost.png",
                         'Data distribution', 'IO cost', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    range_sizes = [0.0006, 0.0025, 0.01, 0.04, 0.16]
    range_size_len = len(range_sizes)
    range_query_time_nyct = [[competitor_names[j], [result.iloc[73 + i][2 + j] * 1000 for i in range(range_size_len)]]
                             for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_time_nyct, output_path + "/range_query_time_nyct.png",
               'Query range size (%)', 'Query time (ms)', competitor_colors, competitor_markers,
               [0.95, 0.15, 0.97, 0.15], False)
    range_query_io_cost_nyct = [[competitor_names[j], [result.iloc[79 + i][2 + j] for i in range(range_size_len)]]
                                for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_io_cost_nyct, output_path + "/range_query_io_cost_nyct.png",
               'Query range size (%)', 'IO cost', competitor_colors, competitor_markers, [0.95, 0.135, 0.97, 0.15],
               False)
    # knn query
    knn_query_time = [[competitor_names[j], [result.iloc[26 + i * 32][2 + j] * 1000 for i in range(dataset_len)]]
                      for j in range(competitor_len)]
    plot_group_histogram(datasets, knn_query_time, output_path + "/knn_query_time.png",
                         'Data distribution', 'Query time (ms)', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    knn_query_io_cost = [[competitor_names[j], [result.iloc[32 + i * 32][2 + j] for i in range(dataset_len)]]
                         for j in range(competitor_len)]
    plot_group_histogram(datasets, knn_query_io_cost, output_path + "/knn_query_io_cost.png",
                         'Data distribution', 'IO cost', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    knn_sizes = [4, 8, 16, 32, 64]
    knn_size_len = len(knn_sizes)
    knn_query_time_nyct = [[competitor_names[j], [result.iloc[85 + i][2 + j] * 1000 for i in range(knn_size_len)]]
                           for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_time_nyct, output_path + "/knn_query_time_nyct.png",
               "${k}$", 'Query time (ms)', competitor_colors, competitor_markers, [0.973, 0.135, 0.97, 0.15], False)
    knn_query_io_cost_nyct = [[competitor_names[j], [result.iloc[91 + i][2 + j] for i in range(knn_size_len)]]
                              for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_io_cost_nyct, output_path + "/knn_query_io_cost_nyct.png",
               "${k}$", 'IO cost', competitor_colors, competitor_markers, [0.975, 0.135, 0.97, 0.15], False)


def plot_compare_update(input_path, output_path, names, colors, markers):
    xls = pd.ExcelFile(input_path)
    # update
    result = pd.ExcelFile.parse(xls, sheet_name='update', header=None)
    update_point_percents = [10, 20, 30, 40, 50]
    update_point_percent_len = len(update_point_percents)
    competitor_ids = [0, 1, 2, 4, 5, 9]
    competitor_colors = [colors[j] for j in competitor_ids]
    competitor_markers = [markers[j] for j in competitor_ids]
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    # update time
    update_time = [[names[j], [np.average(result.iloc[3 + i * 11][1 + update_point_percent_len * j:
                                                                  1 + update_point_percent_len * (j + 1)])
                               for i in range(3)]]
                   for j in competitor_ids]
    plot_group_histogram(datasets, update_time, output_path + "/update_time.png",
                         'Data distribution', 'Update time (s)', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    # update time in nyct
    update_time_nyct = [[names[j], result.iloc[25][1 + update_point_percent_len * j:
                                                   1 + update_point_percent_len * (j + 1)].tolist()]
                        for j in competitor_ids]
    plot_lines(update_point_percents, update_time_nyct, output_path + "/update_time_nyct.png",
               'Updated points (%)', 'Update time (s)', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # point query in nyct
    update_point_query_time_nyct = [[names[j], (result.iloc[28][1 + update_point_percent_len * j:
                                                                1 + update_point_percent_len * (j + 1)]
                                                * 1000000).tolist()]
                                    for j in competitor_ids]
    plot_lines(update_point_percents, update_point_query_time_nyct, output_path + "/update_point_query_time_nyct.png",
               'Updated points (%)', 'Query time (μs)', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    update_point_query_io_cost_nyct = [[names[j], (result.iloc[29][1 + update_point_percent_len * j:
                                                                   1 + update_point_percent_len * (
                                                                           j + 1)]).tolist()]
                                       for j in competitor_ids]
    plot_lines(update_point_percents, update_point_query_io_cost_nyct,
               output_path + "/update_point_query_io_cost_nyct.png",
               'Updated points (%)', 'IO cost', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # range query in nyct
    update_range_query_time_nyct = [[names[j], (result.iloc[30][1 + update_point_percent_len * j:
                                                                1 + update_point_percent_len * (j + 1)]
                                                * 1000).tolist()]
                                    for j in competitor_ids]
    plot_lines(update_point_percents, update_range_query_time_nyct, output_path + "/update_range_query_time_nyct.png",
               'Updated points (%)', 'Query time (ms)', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    update_range_query_io_cost_nyct = [[names[j], (result.iloc[31][1 + update_point_percent_len * j:
                                                                   1 + update_point_percent_len * (
                                                                           j + 1)]).tolist()]
                                       for j in competitor_ids]
    plot_lines(update_point_percents, update_range_query_io_cost_nyct,
               output_path + "/update_range_query_io_cost_nyct.png",
               'Updated points (%)', 'IO cost', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # knn query in nyct
    update_knn_query_time_nyct = [[names[j], (result.iloc[32][1 + update_point_percent_len * j:
                                                              1 + update_point_percent_len * (j + 1)]
                                              * 1000).tolist()]
                                  for j in competitor_ids]
    plot_lines(update_point_percents, update_knn_query_time_nyct, output_path + "/update_knn_query_time_nyct.png",
               'Updated points (%)', 'Query time (ms)', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    update_knn_query_io_cost_nyct = [[names[j], (result.iloc[33][1 + update_point_percent_len * j:
                                                                 1 + update_point_percent_len * (
                                                                         j + 1)]).tolist()]
                                     for j in competitor_ids]
    plot_lines(update_point_percents, update_knn_query_io_cost_nyct, output_path + "/update_knn_query_io_cost_nyct.png",
               'Updated points (%)', 'IO cost', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # update slbrin variants
    competitor_ids = [6, 7, 8, 9]
    competitor_colors = [colors[j] for j in competitor_ids]
    competitor_markers = [markers[j] for j in competitor_ids]
    # update time
    update_time = [[names[j], [np.average(result.iloc[3 + i * 11][1 + update_point_percent_len * j:
                                                                  1 + update_point_percent_len * (j + 1)])
                               for i in range(3)]]
                   for j in competitor_ids]
    plot_group_histogram(datasets, update_time, output_path + "/update_time_slbrin.png",
                         'Data distribution', 'Update time (s)', competitor_colors, [0.99, 0.135, 0.97, 0.15], False)
    # update time in nyct
    update_time_nyct = [[names[j], result.iloc[25][1 + update_point_percent_len * j:
                                                   1 + update_point_percent_len * (j + 1)].tolist()]
                        for j in competitor_ids]
    plot_lines(update_point_percents, update_time_nyct, output_path + "/update_time_nyct_slbrin.png",
               'Updated points (%)', 'Update time (s)', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15],
               False)
    # point query in nyct
    update_point_query_time_nyct = [[names[j], (result.iloc[28][1 + update_point_percent_len * j:
                                                                1 + update_point_percent_len * (j + 1)]
                                                * 1000000).tolist()]
                                    for j in competitor_ids]
    plot_lines(update_point_percents, update_point_query_time_nyct,
               output_path + "/update_point_query_time_nyct_slbrin.png",
               'Updated points (%)', 'Query time (μs)', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    update_point_query_io_cost_nyct = [[names[j], (result.iloc[29][1 + update_point_percent_len * j:
                                                                   1 + update_point_percent_len * (
                                                                           j + 1)]).tolist()]
                                       for j in competitor_ids]
    plot_lines(update_point_percents, update_point_query_io_cost_nyct,
               output_path + "/update_point_query_io_cost_nyct_slbrin.png",
               'Updated points (%)', 'IO cost', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # range query in nyct
    update_range_query_time_nyct = [[names[j], (result.iloc[30][1 + update_point_percent_len * j:
                                                                1 + update_point_percent_len * (j + 1)]
                                                * 1000).tolist()]
                                    for j in competitor_ids]
    plot_lines(update_point_percents, update_range_query_time_nyct,
               output_path + "/update_range_query_time_nyct_slbrin.png",
               'Updated points (%)', 'Query time (ms)', competitor_colors, competitor_markers,
               [0.976, 0.14, 0.97, 0.15], False)
    update_range_query_io_cost_nyct = [[names[j], (result.iloc[31][1 + update_point_percent_len * j:
                                                                   1 + update_point_percent_len * (
                                                                           j + 1)]).tolist()]
                                       for j in competitor_ids]
    plot_lines(update_point_percents, update_range_query_io_cost_nyct,
               output_path + "/update_range_query_io_cost_nyct_slbrin.png",
               'Updated points (%)', 'IO cost', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # knn query in nyct
    update_knn_query_time_nyct = [[names[j], (result.iloc[32][1 + update_point_percent_len * j:
                                                              1 + update_point_percent_len * (j + 1)]
                                              * 1000).tolist()]
                                  for j in competitor_ids]
    plot_lines(update_point_percents, update_knn_query_time_nyct,
               output_path + "/update_knn_query_time_nyct_slbrin.png",
               'Updated points (%)', 'Query time (ms)', competitor_colors, competitor_markers,
               [0.976, 0.15, 0.97, 0.15], False)
    update_knn_query_io_cost_nyct = [[names[j], (result.iloc[33][1 + update_point_percent_len * j:
                                                                 1 + update_point_percent_len * (
                                                                         j + 1)]).tolist()]
                                     for j in competitor_ids]
    plot_lines(update_point_percents, update_knn_query_io_cost_nyct,
               output_path + "/update_knn_query_io_cost_nyct_slbrin.png",
               'Updated points (%)', 'IO cost', competitor_colors, competitor_markers,
               [0.976, 0.135, 0.97, 0.15], False)
    # update slbrin/zm model avg err
    competitor_ids = [4, 5, 6, 7, 8, 9]
    competitor_len = len(competitor_ids)
    competitor_names = [names[j] for j in competitor_ids]
    competitor_colors = [colors[j] for j in competitor_ids]
    competitor_markers = [markers[j] for j in competitor_ids]
    update_err_nyct_slbrin = [[competitor_names[j], (result.iloc[54 + j][1:1 + update_point_percent_len]).tolist()]
                              for j in range(competitor_len)]
    plot_lines(update_point_percents, update_err_nyct_slbrin, output_path + "/update_err_nyct_slbrin.png",
               'Updated points (%)', 'Error bounds', competitor_colors, competitor_markers, [0.976, 0.135, 0.97, 0.15],
               False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # 1. 实验日志转csv，把log文件夹里的所有非csv文件转成csv
    # input_path = "./log"
    # output_path = "./log"
    # log_to_csv(input_path, output_path)
    # 2. 绘制结果图
    names = ['RT', 'PRQT', 'BRINS', 'ZM-NR', 'ZM', 'LISA', 'SLBRIN-SCR', 'SLBRIN-MCR', 'SLBRIN-RM', 'SLBRIN']
    colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#FCE166', '#FCE166', '#86B2C5',
              '#FADEA7', '#E57373', '#F53935', '#B71C1C']
    markers = ['v', 's', 'p', '*', '*', 'x', 'o', 'o', 'o', 'o']
    input_path = "./table/result_slbrin.xlsx"
    output_path = "./result_ijgi"
    plot_grid_search_tn_ts(input_path, output_path)
    plot_compare_build_query(input_path, output_path, names, colors, markers)
    plot_compare_update(input_path, output_path, names, colors, markers)
