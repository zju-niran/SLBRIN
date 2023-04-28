import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rcParams
from matplotlib.ticker import MultipleLocator

config = {
    "font.size": 20,
    "font.family": 'serif',
    "mathtext.fontset": 'stix',
    "font.serif": ['FangSong'],
}
rcParams.update(config)


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
    plt.xticks(group_x, x, fontproperties='Times New Roman', fontsize=font_size)
    plt.yticks(fontproperties='Times New Roman', fontsize=font_size)
    plt.yscale("log")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, ncol=2, columnspacing=1, handletextpad=0.3, fontsize=font_size)
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_group_histogram(x, y_list, model_path, x_title, y_title, color_list, adjust, is_legend=True,
                         legend_pos='best', is_log=True, ncol=2):
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
    plt.xticks(x_arange, x, fontproperties='Times New Roman', fontsize=font_size)
    plt.yticks(fontproperties='Times New Roman', fontsize=font_size)
    if is_log:
        plt.yscale("log")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, ncol=ncol, columnspacing=1, handletextpad=0.3, fontsize=font_size)
    plt.savefig(model_path)
    plt.close()


def plot_histogram_and_line(x, y1, y2, model_path, x_title, y_title1, y_title2, y_label1, y_label2,
                            color1, color2, adjust, y_lim1=None, y_lim2=None, is_legend=True, legend_pos=(1, 1)):
    marker_size = 15
    tn_x = list(range(5))
    fig = plt.figure()
    # 坐标轴1
    ax1 = fig.add_subplot(111)
    plt.xticks(tn_x, x, fontproperties='Times New Roman')
    plt.xlabel(x_title)
    plt.yticks(fontproperties='Times New Roman')
    ax1.bar(tn_x, y1, color=color1, label=y_label1, width=0.6)
    ax1.set_ylabel(y_title1)
    if y_lim1:
        ax1.yaxis.set_major_locator(MultipleLocator(y_lim1[2]))
        plt.ylim(y_lim1[0], y_lim1[1])
    # 坐标轴2
    ax2 = ax1.twinx()
    ax2.plot(tn_x, y2, color=color2, label=y_label2, linestyle='-', linewidth=4, marker='o',
             markersize=marker_size)
    ax2.set_ylabel(y_title2)
    plt.yticks(fontproperties='Times New Roman')
    ax2.tick_params(axis='y')
    if y_lim2:
        ax2.yaxis.set_major_locator(MultipleLocator(y_lim2[2]))
        plt.ylim(y_lim2[0], y_lim2[1])
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    plt.margins(x=0)
    if is_legend:
        fig.legend(loc=1, bbox_to_anchor=legend_pos, bbox_transform=ax1.transAxes, frameon=False,
                   ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_grid_search_slbrin(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    # TN
    tn_result = pd.ExcelFile.parse(xls, sheet_name='slbrin', header=None)
    tn_x = ['5000', '7500', '10000', '15000', '20000']
    tn_build_time = [tn_result.iloc[66][1 + i] / 1000 for i in range(5)]
    tn_index_size = [tn_result.iloc[68][1 + i] / 1024 / 1024 for i in range(5)]
    tn_error_range = [tn_result.iloc[71][1 + i] / 100 for i in range(5)]
    tn_error_ratio = [tn_result.iloc[72][1 + i] * 100 for i in range(5)]
    tn_query_time = [sum([tn_result.iloc[75 + i][1 + j] for i in range(5)]) / 5 * 100000 for j in range(5)]
    tn_query_io_cost = [sum([tn_result.iloc[80 + i][1 + j] for i in range(5)]) / 5 for j in range(5)]

    # 5000 6000 8000 10000 20000转为均匀分布
    def fun(l):
        l[1] = (l[1] + l[2]) / 2
        l[2] = l[3]
        l[3] = (l[2] + l[4]) / 2

    fun(tn_build_time)
    fun(tn_index_size)
    fun(tn_error_range)
    fun(tn_error_ratio)
    fun(tn_query_time)
    fun(tn_query_io_cost)
    tn_x_title = "$\mathit{TN}$"
    plot_histogram_and_line(tn_x, tn_index_size, tn_build_time, output_path + "/tn1.png",
                            tn_x_title, "索引体积（$\mathrm{MB}$）", "构建时间（$\mathrm{1000s}$）", "索引体积", "构建时间",
                            color1, color2, [0.89, 0.11, 0.97, 0.15], [0, 15, 3], [0, 10, 2], True, (1, 1))
    plot_histogram_and_line(tn_x, tn_error_range, tn_error_ratio, output_path + "/tn2.png",
                            tn_x_title, "误差范围（$\mathrm{×100}$）", "误差率（$\mathrm{\%}$）", "误差范围", "误差率",
                            color1, color2, [0.88, 0.11, 0.97, 0.15], [0, 10, 2], [3.4, 3.9, 0.1], True, (0.85, 1))
    plot_histogram_and_line(tn_x, tn_query_io_cost, tn_query_time, output_path + "/tn3.png",
                            tn_x_title, "检索$\mathrm{IO}$", "检索时间（$\mathrm{0.01ms}$）", "检索$\mathrm{IO}$", "检索时间",
                            color1, color2, [0.89, 0.11, 0.97, 0.15], [50, 70, 5], [23, 27, 1], True, (0.85, 1))


def plot_compare_build(input_path, output_path, names, colors):
    xls = pd.ExcelFile(input_path)
    result = pd.ExcelFile.parse(xls, sheet_name='build_query', header=None)
    competitor_ids = [0, 1, 2, 4, 5, 9]
    competitor_names = [names[j] for j in competitor_ids]
    competitor_colors = [colors[j] for j in competitor_ids]
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # index size
    index_structure_sizes = [[competitor_names[j],
                              [result.iloc[3 + i * 32][2 + j] / 1024 / 1024 for i in range(dataset_len)]]
                             for j in range(competitor_len)]
    plot_group_histogram(datasets, index_structure_sizes, output_path + "/build1.png",
                         '数据集', '索引体积（$\mathrm{MB}$）', competitor_colors, [0.99, 0.145, 0.97, 0.15], False)
    # build time
    build_times = [[competitor_names[j], [result.iloc[2 + i * 32][2 + j] for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, build_times, output_path + "/build2.png",
                         '数据集', '构建时间（$\mathrm{s}$）', competitor_colors, [0.99, 0.13, 0.97, 0.15], False)


def plot_compare_error(input_path, output_path, names, colors):
    xls = pd.ExcelFile(input_path)
    result = pd.ExcelFile.parse(xls, sheet_name='build_query', header=None)
    competitor_ids = [4, 5, 9]
    competitor_names = [names[j] for j in competitor_ids]
    competitor_colors = [colors[j] for j in competitor_ids]
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # err
    error_range = [[competitor_names[j], [result.iloc[5 + i * 32][5 + j] / 100 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, error_range, output_path + "/error1.png", '数据集', '误差范围（$\mathrm{×100}$）',
                         competitor_colors, [0.99, 0.09, 0.97, 0.15], is_legend=False, is_log=False, ncol=1)
    error_ratio = [[competitor_names[j], [result.iloc[6 + i * 32][5 + j] * 100 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, error_ratio, output_path + "/error2.png", '数据集', "误差率（$\mathrm{\%}$）",
                         competitor_colors, [0.99, 0.11, 0.97, 0.15], is_legend=False, is_log=False)


def plot_compare_query(input_path, output_path, names, colors, markers):
    xls = pd.ExcelFile(input_path)
    result = pd.ExcelFile.parse(xls, sheet_name='build_query', header=None)
    competitor_ids = [0, 1, 2, 4, 5, 9]
    competitor_names = [names[j] for j in competitor_ids]
    competitor_colors = [colors[j] for j in competitor_ids]
    competitor_markers = [markers[j] for j in competitor_ids]
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # point query
    point_query_io_cost = [[competitor_names[j], [result.iloc[8 + i * 32][2 + j] for i in range(dataset_len)]]
                           for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query_io_cost, output_path + "/point_query1.png", '数据集', '检索$\mathrm{IO}$',
                         competitor_colors, [0.99, 0.13, 0.97, 0.15], False)
    point_query_time = [[competitor_names[j], [result.iloc[7 + i * 32][2 + j] * 1000000 for i in range(dataset_len)]]
                        for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query_time, output_path + "/point_query2.png", '数据集', '检索时间（$\mathrm{μs}$）',
                         competitor_colors, [0.99, 0.13, 0.97, 0.15], False)
    # range query
    range_query_io_cost = [[competitor_names[j], [result.iloc[20 + i * 32][2 + j] for i in range(dataset_len)]]
                           for j in range(competitor_len)]
    plot_group_histogram(datasets, range_query_io_cost, output_path + "/range_query1.png", '数据集', '检索$\mathrm{IO}$',
                         competitor_colors, [0.99, 0.13, 0.97, 0.15], False, 'upper left')
    range_query_time = [[competitor_names[j], [result.iloc[14 + i * 32][2 + j] * 1000 for i in range(dataset_len)]]
                        for j in range(competitor_len)]
    plot_group_histogram(datasets, range_query_time, output_path + "/range_query2.png", '数据集', '检索时间（$\mathrm{ms}$）',
                         competitor_colors, [0.99, 0.13, 0.97, 0.15], False)
    range_sizes = [0.0006, 0.0025, 0.01, 0.04, 0.16]
    range_size_len = len(range_sizes)
    range_query_io_cost_nyct = [[competitor_names[j], [result.iloc[79 + i][2 + j] for i in range(range_size_len)]]
                                for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_io_cost_nyct, output_path + "/range_query3.png",
               '查询框大小（$\mathrm{\%}$）', '检索$\mathrm{IO}$',
               competitor_colors, competitor_markers, [0.95, 0.13, 0.97, 0.15], False, 'upper left')
    range_query_time_nyct = [[competitor_names[j], [result.iloc[73 + i][2 + j] * 1000 for i in range(range_size_len)]]
                             for j in range(competitor_len)]
    plot_lines(range_sizes, range_query_time_nyct, output_path + "/range_query4.png",
               '查询框大小（$\mathrm{\%}$）', '检索时间（$\mathrm{ms}$）',
               competitor_colors, competitor_markers, [0.95, 0.145, 0.97, 0.15], False)
    # knn query
    knn_query_io_cost = [[competitor_names[j], [result.iloc[32 + i * 32][2 + j] for i in range(dataset_len)]]
                         for j in range(competitor_len)]
    plot_group_histogram(datasets, knn_query_io_cost, output_path + "/knn_query1.png", '数据集', '检索$\mathrm{IO}$',
                         competitor_colors, [0.99, 0.13, 0.97, 0.15], False)
    knn_query_time = [[competitor_names[j], [result.iloc[26 + i * 32][2 + j] * 1000 for i in range(dataset_len)]]
                      for j in range(competitor_len)]
    plot_group_histogram(datasets, knn_query_time, output_path + "/knn_query2.png", '数据集', '检索时间（$\mathrm{ms}$）',
                         competitor_colors, [0.99, 0.13, 0.97, 0.15], False)
    knn_sizes = [4, 8, 16, 32, 64]
    knn_size_len = len(knn_sizes)
    knn_query_io_cost_nyct = [[competitor_names[j], [result.iloc[91 + i][2 + j] for i in range(knn_size_len)]]
                              for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_io_cost_nyct, output_path + "/knn_query3.png", "${k}$", '检索$\mathrm{IO}$',
               competitor_colors, competitor_markers, [0.975, 0.13, 0.97, 0.15], False, 'upper left')
    knn_query_time_nyct = [[competitor_names[j], [result.iloc[85 + i][2 + j] for i in range(knn_size_len)]]
                           for j in range(competitor_len)]
    plot_lines(knn_sizes, knn_query_time_nyct, output_path + "/knn_query4.png", "${k}$", '检索时间（$\mathrm{ms}$）',
               competitor_colors, competitor_markers, [0.973, 0.15, 0.97, 0.15], False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    names = ['$\mathrm{RT}$', '$\mathrm{PRQT}$', '$\mathrm{BRINS}$',
             '$\mathrm{ZM-NR}$', '$\mathrm{ZM}$', '$\mathrm{LISA}$',
             '$\mathrm{SLBRIN-SCR}$', '$\mathrm{SLBRIN-MCR}$', '$\mathrm{SLBRIN-RM}$', '$\mathrm{SLBRIN}$']
    colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#FCE166', '#FCE166', '#86B2C5',
              '#FADEA7', '#E57373', '#F53935', '#B71C1C']
    markers = ['v', 's', 'p', '*', '*', 'x', 'o', 'o', 'o', 'o']
    input_path = "./table/result_slbrin.xlsx"
    output_path = "./result_slbrin"
    plot_grid_search_slbrin(input_path, output_path)
    plot_compare_build(input_path, output_path, names, colors)
    plot_compare_error(input_path, output_path, names, colors)
    plot_compare_query(input_path, output_path, names, colors, markers)
