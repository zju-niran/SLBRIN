import csv
import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches, lines


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


def plot_group_histogram(x, y_list, model_path, x_title, y_title, color_list):
    """
    分组直方图，y轴表示为10的幂次
    :param x: 组名list
    :param y_list: 组对应的值list
    :param model_path: 保存路径
    :param x_title: x轴标题
    :param y_title: y轴标题
    :param color_list: 组内各柱的颜色
    """
    font_size = 12
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
    plt.legend(loc='best', frameon=False, fontsize=font_size, ncol=2)
    plt.savefig(model_path)
    plt.close()


def plot_group_histogram_and_line(x, y_list, model_path, x_title, y_title, line_title, color_list):
    font_size = 12
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
    # plt.yscale("log")
    plt.legend(handles=group_member_labels + overlay_line_labels, loc='best', frameon=False, fontsize=font_size, ncol=2)
    plt.savefig(model_path)
    plt.close()


def plot_lines(x, y_list, model_path, x_title, y_title, color_list, marker_list):
    font_size = 12
    marker_size = 10
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
    plt.legend(loc='best', frameon=False, fontsize=font_size, ncol=2)
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_result(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    result = pd.ExcelFile.parse(xls, sheet_name='build_query', header=None)
    competitors = ['RT', 'PRQT', 'KDT', 'BRINS', 'ZM', 'SBRIN']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#FCE166', '#86B2C5', '#E63025']
    competitor_markers = ['v', 's', 'p', '*', 'x', 'o']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitors)
    dataset_len = len(datasets)
    # construction time
    construction_times = [[competitors[j], [result.iloc[2 + i * 18][2 + j] for i in range(dataset_len)]]
                          for j in range(competitor_len)]
    plot_group_histogram(datasets, construction_times, output_path + "/construction_time.png",
                         'Data distribution', 'Construction time (s)', competitor_colors)
    # index size
    index_structure_sizes = [[competitors[j],
                              [result.iloc[3 + i * 18][2 + j] / 1024 / 1024 for i in range(dataset_len)]]
                             for j in range(competitor_len)]
    plot_group_histogram(datasets, index_structure_sizes, output_path + "/index_structure_size.png",
                         'Data distribution', 'Index structure size (MB)', competitor_colors)
    index_sizes = [[competitors[j],
                    [result.iloc[3 + i * 18][2 + j] / 1024 / 1024 for i in range(dataset_len)],
                    [result.iloc[4 + i * 18][2 + j] / 1024 / 1024 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram_and_line(datasets, index_sizes, output_path + "/index_size.png",
                                  'Data distribution', 'Index size (MB)', 'Index entry size', competitor_colors)
    # io cost
    io_costs = [[competitors[j], [result.iloc[5 + i * 18][2 + j] for i in range(dataset_len)]]
                for j in range(competitor_len)]
    plot_group_histogram(datasets, io_costs, output_path + "/io_cost.png",
                         'Data distribution', 'IO cost', competitor_colors)
    # point query
    point_query = [[competitors[j], [result.iloc[6 + i * 18][2 + j] * 1000000 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query, output_path + "/point_query.png",
                         'Data distribution', 'Average query time (μs)', competitor_colors)
    # range query
    range_query = [[competitors[j], [result.iloc[12 + i * 18][2 + j] * 1000 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, range_query, output_path + "/range_query.png",
                         'Data distribution', 'Average query time (ms)', competitor_colors)
    range_sizes = [0.0006, 0.0025, 0.01, 0.04, 0.16]
    range_size_len = len(range_sizes)
    range_query_uniform = [[competitors[j], [result.iloc[7 + i][2 + j] * 1000 for i in range(range_size_len)]]
                           for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_uniform, output_path + "/range_query_uniform.png",
               'Query window size (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    range_query_normal = [[competitors[j], [result.iloc[25 + i][2 + j] * 1000 for i in range(range_size_len)]]
                          for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_normal, output_path + "/range_query_normal.png",
               'Query windowe size (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    range_query_nyct = [[competitors[j], [result.iloc[43 + i][2 + j] * 1000 for i in range(range_size_len)]]
                        for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_nyct, output_path + "/range_query_nyct.png",
               'Query window size (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    # knn query
    knn_query = [[competitors[j], [result.iloc[18 + i * 18][2 + j] * 1000 for i in range(dataset_len)]]
                 for j in range(competitor_len)]
    plot_group_histogram(datasets, knn_query, output_path + "/knn_query.png",
                         'Data distribution', 'Average query time (ms)', competitor_colors)
    knn_sizes = [4, 8, 16, 32, 64]
    knn_size_len = len(knn_sizes)
    knn_query_uniform = [[competitors[j], [result.iloc[13 + i][2 + j] for i in range(knn_size_len)]]
                         for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_uniform, output_path + "/knn_query_uniform.png",
               'k', 'Average query time (μs)', competitor_colors, competitor_markers)
    knn_query_normal = [[competitors[j], [result.iloc[31 + i][2 + j] for i in range(knn_size_len)]]
                        for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_normal, output_path + "/knn_query_normal.png",
               'k', 'Average query time (μs)', competitor_colors, competitor_markers)
    knn_query_nyct = [[competitors[j], [result.iloc[49 + i][2 + j] for i in range(knn_size_len)]]
                      for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_nyct, output_path + "/knn_query_nyct.png",
               'k', 'Average query time (μs)', competitor_colors, competitor_markers)
    # update
    result = pd.ExcelFile.parse(xls, sheet_name='update_optimized', header=None)
    insert_point_percents = [10, 20, 30, 40, 50]
    insert_point_percent_len = len(insert_point_percents)
    competitors = ['RT', 'PRQT', 'KDT', 'BRINS', 'ZM', 'SBRIN-SCR', 'SBRIN-MCR', 'SBRIN-RM', 'SBRIN']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#FCE166', '#86B2C5', '#E63025']
    competitor_markers = ['v', 's', 'p', '*', 'x', 'o']
    competitor_ids = [0, 1, 2, 3, 4, 8]
    # update nyct
    update_time_nyct = [[competitors[j], result.iloc[21][1 + insert_point_percent_len * j:
                                                         1 + insert_point_percent_len * (j + 1)].tolist()]
                        for j in competitor_ids]
    plot_lines(insert_point_percents, update_time_nyct, output_path + "/update_time_nyct.png",
               'Inserted points (%)', 'Update time (s)', competitor_colors, competitor_markers)
    update_size_nyct = [[competitors[j], (result.iloc[22][1 + insert_point_percent_len * j:
                                                          1 + insert_point_percent_len * (j + 1)]
                                          / 1024 / 1024).tolist()]
                        for j in competitor_ids]
    plot_lines(insert_point_percents, update_size_nyct, output_path + "/update_size_nyct.png",
               'Inserted points (%)', 'Index structure size (MB)', competitor_colors, competitor_markers)
    update_io_nyct = [[competitors[j], result.iloc[24][1 + insert_point_percent_len * j:
                                                       1 + insert_point_percent_len * (j + 1)].tolist()]
                      for j in competitor_ids]
    plot_lines(insert_point_percents, update_io_nyct, output_path + "/update_io_nyct.png",
               'Inserted points (%)', 'IO cost', competitor_colors, competitor_markers)
    update_point_query_nyct = [[competitors[j], (result.iloc[25][1 + insert_point_percent_len * j:
                                                                 1 + insert_point_percent_len * (j + 1)]
                                                 * 1000000).tolist()]
                               for j in competitor_ids]
    plot_lines(insert_point_percents, update_point_query_nyct, output_path + "/update_point_query_nyct.png",
               'Inserted points (%)', 'Average query time (μs)', competitor_colors, competitor_markers)
    update_range_query_nyct = [[competitors[j], (result.iloc[26][1 + insert_point_percent_len * j:
                                                                 1 + insert_point_percent_len * (j + 1)]
                                                 * 1000).tolist()]
                               for j in competitor_ids]
    plot_lines(insert_point_percents, update_range_query_nyct, output_path + "/update_range_query_nyct.png",
               'Inserted points (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    update_knn_query_nyct = [[competitors[j], (result.iloc[27][1 + insert_point_percent_len * j:
                                                               1 + insert_point_percent_len * (j + 1)]
                                               * 1000).tolist()]
                             for j in competitor_ids]
    plot_lines(insert_point_percents, update_knn_query_nyct, output_path + "/update_knn_query_nyct.png",
               'Inserted points (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    # update sbrin variants
    competitor_colors = ['#E6B597', '#FF962A', '#EA7D80', '#E63025']
    competitor_markers = ['o', 'o', 'o', 'o']
    competitor_ids = [5, 6, 7, 8]
    update_time_nyct_sbrin = [[competitors[j], result.iloc[21][1 + insert_point_percent_len * j:
                                                         1 + insert_point_percent_len * (j + 1)].tolist()]
                        for j in competitor_ids]
    plot_lines(insert_point_percents, update_time_nyct_sbrin, output_path + "/update_time_nyct_sbrin.png",
               'Inserted points (%)', 'Update time (s)', competitor_colors, competitor_markers)
    update_size_nyct_sbrin = [[competitors[j], (result.iloc[22][1 + insert_point_percent_len * j:
                                                          1 + insert_point_percent_len * (j + 1)]
                                          / 1024 / 1024).tolist()]
                        for j in competitor_ids]
    plot_lines(insert_point_percents, update_size_nyct_sbrin, output_path + "/update_size_nyct_sbrin.png",
               'Inserted points (%)', 'Index structure size (MB)', competitor_colors, competitor_markers)
    update_io_nyct_sbrin = [[competitors[j], result.iloc[24][1 + insert_point_percent_len * j:
                                                       1 + insert_point_percent_len * (j + 1)].tolist()]
                      for j in competitor_ids]
    plot_lines(insert_point_percents, update_io_nyct_sbrin, output_path + "/update_io_nyct_sbrin.png",
               'Inserted points (%)', 'IO cost', competitor_colors, competitor_markers)
    update_point_query_nyct_sbrin = [[competitors[j], (result.iloc[25][1 + insert_point_percent_len * j:
                                                                 1 + insert_point_percent_len * (j + 1)]
                                                 * 1000000).tolist()]
                               for j in competitor_ids]
    plot_lines(insert_point_percents, update_point_query_nyct_sbrin, output_path + "/update_point_query_nyct_sbrin.png",
               'Inserted points (%)', 'Average query time (μs)', competitor_colors, competitor_markers)
    update_range_query_nyct_sbrin = [[competitors[j], (result.iloc[26][1 + insert_point_percent_len * j:
                                                                 1 + insert_point_percent_len * (j + 1)]
                                                 * 1000).tolist()]
                               for j in competitor_ids]
    plot_lines(insert_point_percents, update_range_query_nyct_sbrin, output_path + "/update_range_query_nyct_sbrin.png",
               'Inserted points (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    update_knn_query_nyct_sbrin = [[competitors[j], (result.iloc[27][1 + insert_point_percent_len * j:
                                                               1 + insert_point_percent_len * (j + 1)]
                                               * 1000).tolist()]
                             for j in competitor_ids]
    plot_lines(insert_point_percents, update_knn_query_nyct_sbrin, output_path + "/update_knn_query_nyct_sbrin.png",
               'Inserted points (%)', 'Average query time (ms)', competitor_colors, competitor_markers)
    # update sbrin/zm model avg err
    competitors = ['ZM', 'SBRIN-SCR', 'SBRIN-MCR', 'SBRIN-RM', 'SBRIN']
    competitor_colors = ['#86B2C5', '#FADEA7', '#FF962A', '#EA7D80', '#E63025']
    competitor_markers = ['x', 'o', 'o', 'o', 'o']
    competitor_len = len(competitors)
    update_err_nyct_sbrin = [[competitors[j], (result.iloc[45 + j][1:1 + insert_point_percent_len]).tolist()]
                       for j in range(competitor_len)]
    plot_lines(insert_point_percents, update_err_nyct_sbrin, output_path + "/update_err_nyct_sbrin.png",
               'Inserted points (%)', 'Error bounds', competitor_colors, competitor_markers)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # 1. 实验日志转csv，把log文件夹里的所有非csv文件转成csv
    input_path = "./log"
    output_path = "./log"
    log_to_csv(input_path, output_path)
    # 2. 绘制point query结果图
    input_path = "./table/result.xlsx"
    output_path = "./png"
    plot_result(input_path, output_path)
