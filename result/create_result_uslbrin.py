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


def plot_lines(x, y_list, model_path, x_title, y_title, color_list, marker_list, adjust, x_lim=None, y_lim=None,
               is_legend=False, legend_pos='best', is_log=False):
    marker_size = 15
    if x_lim:
        major_locator = x_lim[2]
        minor_locator = x_lim[3]
        ax = plt.axes()
        ax.xaxis.set_major_locator(MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_locator))
        plt.xlim(x_lim[0], x_lim[1])
    if is_log:
        plt.yscale("log")
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    for i in range(len(y_list)):
        plt.plot(x, y_list[i][1], label=y_list[i][0],
                 color=color_list[i], marker=marker_list[i], markersize=marker_size)

    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, ncol=2, columnspacing=1, handletextpad=0.3)
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_T_lines(x, y_list, model_path, x_title, y_title, color_list, marker_list, adjust, is_legend=False,
                 legend_pos='best', is_log=False):
    marker_size = 10
    major_locator = 7
    minor_locator = 1
    ax = plt.axes()
    ax.xaxis.set_major_locator(MultipleLocator(major_locator))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_locator))
    for i in range(len(y_list)):
        # 平均值平滑
        # for j in range(1, len(y_list[i][1]) - 1):
        #     y_list[i][1][j] = (y_list[i][1][j + 1] + y_list[i][1][j - 1]) / 2
        if color_list:
            plt.plot(x, y_list[i][1], label=y_list[i][0],
                     color=color_list[i], marker=marker_list[i], markersize=marker_size)
        else:
            plt.plot(x, y_list[i][1])
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlim(0, 27)
    if is_log:
        plt.yscale("log")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, ncol=1, columnspacing=1, handletextpad=0.3)
    plt.margins(x=0)
    plt.savefig(model_path)
    plt.close()


def plot_histogram_and_line(x, y1, y2, model_path, x_title, y_title1, y_title2, y_label1, y_label2,
                            color1, color2, adjust, y_lim1=None, y_lim2=None,
                            is_legend=False, legend_pos=(1, 1), is_log1=False, is_log2=False):
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
    if is_log1:
        plt.yscale("log")
    else:
        plt.yscale("linear")
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
    if is_log2:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    plt.margins(x=0)
    if is_legend:
        fig.legend(loc=1, bbox_to_anchor=legend_pos, bbox_transform=ax1.transAxes, frameon=False,
                   ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_line_and_line(x, y1, y2, model_path, x_title, y_title1, y_title2, y_label1, y_label2,
                       color1, color2, adjust, y_lim1=None, y_lim2=None,
                       is_legend=False, legend_pos=(1, 1), is_log1=False, is_log2=False):
    marker_size = 15
    tn_x = list(range(5))
    fig = plt.figure()
    # 坐标轴1
    ax1 = fig.add_subplot(111)
    plt.xticks(tn_x, x, fontproperties='Times New Roman')
    plt.xlabel(x_title)
    plt.yticks(fontproperties='Times New Roman')
    ax1.plot(tn_x, y1, color=color1, label=y_label1, linestyle='-', linewidth=4, marker='s',
             markersize=marker_size)
    ax1.set_ylabel(y_title1)
    if y_lim1:
        ax1.yaxis.set_major_locator(MultipleLocator(y_lim1[2]))
        plt.ylim(y_lim1[0], y_lim1[1])
    if is_log1:
        plt.yscale("log")
    else:
        plt.yscale("linear")
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
    if is_log2:
        plt.yscale("log")
    else:
        plt.yscale("linear")
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    plt.margins(x=0)
    if is_legend:
        fig.legend(loc=1, bbox_to_anchor=legend_pos, bbox_transform=ax1.transAxes, frameon=False,
                   ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_group_histogram(x, y_list, model_path, x_title, y_title, color_list, adjust, y_lim=None,
                         is_legend=False, legend_pos='best', is_log=False):
    x_arange = np.arange(len(x))
    group_member_len = len(y_list)
    width = 0.1
    for i in range(group_member_len):
        plt.bar(x=x_arange - width * group_member_len / 2 + width / 2 * (i * 2 + 1),
                height=y_list[i][1],
                width=width,
                label=y_list[i][0],
                color=color_list[i])
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.xticks(x_arange, x, fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    if is_log:
        plt.yscale("log")
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, ncol=2, columnspacing=1, handletextpad=0.3)
    plt.savefig(model_path)
    plt.close()


def plot_grid_search_tel(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    tel_result = pd.ExcelFile.parse(xls, sheet_name='TEL', header=None)
    tel_x = ['0.8', '0.9', '1.0', '1.1', '1.2']
    tel_x_offset = [0, 1, 2, 3, 4]
    tel_retrain_num = [tel_result.iloc[29][16 + offset] for offset in tel_x_offset]
    tel_retrain_time = [tel_result.iloc[32][16 + offset] * 1000000 for offset in tel_x_offset]
    tel_error_range = [tel_result.iloc[31][16 + offset] / 100 for offset in tel_x_offset]
    tel_error_ratio = [tel_result.iloc[45][16 + offset] * 100 for offset in tel_x_offset]
    tel_update_time = [tel_result.iloc[43][16 + offset] * 1000000 for offset in tel_x_offset]
    tel_query_time = [tel_result.iloc[17][16 + offset] * 1000000 for offset in tel_x_offset]
    tel_x_title = "$\mathit{TEL}$"
    plot_histogram_and_line(tel_x, tel_retrain_num, tel_retrain_time, output_path + "/tel1.png", tel_x_title,
                            "重训练次数", "重训练时间（$\mathrm{μs}$）",
                            "重训练次数", "重训练时间",
                            color1, color2, [0.87, 0.125, 0.97, 0.15], [50, 15000, 1000], [20, 5000, 100], True, (1, 1),
                            True, True)
    plot_histogram_and_line(tel_x, tel_error_range, tel_error_ratio, output_path + "/tel2.png", tel_x_title,
                            "误差范围（$\mathrm{×100}$）", "误差率（$\mathrm{\%}$）",
                            "误差范围", "误差率",
                            color1, color2, [0.855, 0.12, 0.97, 0.15], [3.4, 4.3, 0.2], [0.34, 0.43, 0.02], True,
                            (1, 1))
    plot_line_and_line(tel_x, tel_update_time, tel_query_time, output_path + "/tel3.png", tel_x_title,
                       "更新时间（$\mathrm{μs}$）", "检索时间（$\mathrm{μs}$）",
                       "更新时间", "检索时间",
                       color1, color2, [0.85, 0.16, 0.97, 0.15], [0, 2000, 500], [36.4, 36.85, 0.1], True, (1, 1))


def plot_grid_search_tef(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    tef_result = pd.ExcelFile.parse(xls, sheet_name='TEF', header=None)
    tef_x = ['0', '1', '2', '4', '8']
    tef_x_offset = [0, 1, 2, 3, 4]
    tef_retrain_num = [tef_result.iloc[34][16 + offset] / 100 for offset in tef_x_offset]
    tef_retrain_time = [tef_result.iloc[40][16 + offset] * 1000000 for offset in tef_x_offset]
    tef_pre_cdf_mae = [tef_result.iloc[36][16 + offset] * 100 for offset in tef_x_offset]
    tef_true_cdf_mae = [tef_result.iloc[38][16 + offset] * 100 for offset in tef_x_offset]
    tef_update_time = [tef_result.iloc[43][16 + offset] * 1000000 for offset in tef_x_offset]
    tef_query_time = [tef_result.iloc[17][16 + offset] * 1000000 for offset in tef_x_offset]
    tef_x_title = "$\mathit{TEF}$"
    plot_histogram_and_line(tef_x, tef_retrain_num, tef_retrain_time, output_path + "/tef1.png", tef_x_title,
                            "重训练次数（$\mathrm{×100}$）", "重训练时间（$\mathrm{μs}$）",
                            "重训练次数", "重训练时间",
                            color1, color2, [0.87, 0.09, 0.97, 0.15], [0, 8, 2], [70, 110, 10], True, (1, 1))
    plot_histogram_and_line(tef_x, tef_pre_cdf_mae, tef_true_cdf_mae, output_path + "/tef2.png", tef_x_title,
                            "预测误差（$\mathrm{\%}$）", "真实误差（$\mathrm{\%}$）",
                            "预测误差", "真实误差",
                            color1, color2, [0.835, 0.145, 0.97, 0.15], [3.55, 3.75, 0.05], [3.232, 3.240, 0.002], True,
                            (1, 1))
    plot_line_and_line(tef_x, tef_update_time, tef_query_time, output_path + "/tef3.png", tef_x_title,
                       "更新时间（$\mathrm{μs}$）", "检索时间（$\mathrm{μs}$）",
                       "更新时间", "检索时间",
                       color1, color2, [0.85, 0.14, 0.97, 0.15], [195, 235, 10], [36.3, 37.1, 0.02], True, (1, 1))


def plot_grid_search_ten(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    ten_result = pd.ExcelFile.parse(xls, sheet_name='TEN', header=None)
    ten_x = ['0.8', '0.9', '1.0', '1.1', '1.2']
    ten_x_offset = [0, 1, 2, 3, 4]
    ten_retrain_num = [ten_result.iloc[35][16 + offset] / 100 for offset in ten_x_offset]
    ten_retrain_time = [ten_result.iloc[40][16 + offset] * 1000000 for offset in ten_x_offset]
    ten_pre_max_key_mae = [ten_result.iloc[37][16 + offset] for offset in ten_x_offset]
    ten_true_max_key_mae = [ten_result.iloc[39][16 + offset] for offset in ten_x_offset]
    ten_update_time = [ten_result.iloc[43][16 + offset] * 1000000 for offset in ten_x_offset]
    ten_query_time = [ten_result.iloc[17][16 + offset] * 1000000 for offset in ten_x_offset]
    ten_x_title = "$\mathit{TEN}$"
    plot_histogram_and_line(ten_x, ten_retrain_num, ten_retrain_time, output_path + "/ten1.png", ten_x_title,
                            "重训练次数（$\mathrm{×100}$）", "重训练时间（$\mathrm{μs}$）",
                            "重训练次数", "重训练时间",
                            color1, color2, [0.87, 0.09, 0.97, 0.15], [0, 8, 2], [70, 170, 20], True, (1, 1))
    plot_histogram_and_line(ten_x, ten_pre_max_key_mae, ten_true_max_key_mae, output_path + "/ten2.png", ten_x_title,
                            "预测误差", "真实误差",
                            "预测误差", "真实误差",
                            color1, color2, [0.835, 0.14, 0.97, 0.15], [10, 10.45, 0.1], [15.3, 15.7, 0.1], True,
                            (1, 1))
    plot_line_and_line(ten_x, ten_update_time, ten_query_time, output_path + "/ten3.png", ten_x_title,
                       "更新时间（$\mathrm{μs}$）", "检索时间（$\mathrm{μs}$）",
                       "更新时间", "检索时间",
                       color1, color2, [0.85, 0.14, 0.97, 0.15], [200, 290, 20], [36.35, 36.9, 0.1], True, (1, 1))


def plot_grid_search_rm(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    rm_result = pd.ExcelFile.parse(xls, sheet_name='TCRM', header=None)
    competitor_colors = ['#FCE166', '#B71C1C', '#FCE166', '#B71C1C']
    competitor_markers = ['*', '*', 'o', 'o']
    competitor_names = ['$\mathrm{TSUSLI-NHD}$', '$\mathrm{TSUSLI}$', '$\mathrm{USLBRIN-NHD}$', '$\mathrm{USLBRIN}$']
    competitor_len = len(competitor_names)
    update_time_id_list = list(range(1, 28, 1))
    rm_retrain_avg_time = [[competitor_names[j], [rm_result.iloc[31 + i * 37][14 + j] /
                                                  rm_result.iloc[29 + i * 37][14 + j]
                                                  if rm_result.iloc[29 + i * 37][14 + j] else 0
                                                  for i in range(27)]]
                           for j in range(competitor_len)]
    rm_retrain_num = [[competitor_names[j], [rm_result.iloc[29 + i * 37][14 + j] / 100 for i in range(27)]]
                      for j in range(competitor_len)]
    rm_retrain_time = [[competitor_names[j], [rm_result.iloc[31 + i * 37][14 + j] * 1000000 /
                                              (rm_result.iloc[7 + i * 37][14 + j] +
                                               rm_result.iloc[10 + i * 37][14 + j] +
                                               rm_result.iloc[13 + i * 37][14 + j] +
                                               rm_result.iloc[16 + i * 37][14 + j] +
                                               rm_result.iloc[19 + i * 37][14 + j] +
                                               rm_result.iloc[41 + i * 37][14 + j]) for i in range(27)]]
                       for j in range(competitor_len)]
    rm_update_time = [[competitor_names[j], [(rm_result.iloc[25 + i * 37][14 + j] +
                                              rm_result.iloc[27 + i * 37][14 + j] +
                                              rm_result.iloc[31 + i * 37][14 + j] +
                                              rm_result.iloc[31 + i * 37][14 + j]) * 1000000 /
                                             (rm_result.iloc[37 + i * 37][14 + j] +
                                              rm_result.iloc[10 + i * 37][14 + j] +
                                              rm_result.iloc[13 + i * 37][14 + j] +
                                              rm_result.iloc[16 + i * 37][14 + j] +
                                              rm_result.iloc[19 + i * 37][14 + j] +
                                              rm_result.iloc[41 + i * 37][14 + j]) for i in range(27)]]
                      for j in range(competitor_len)]
    rm_error_ratio = [[competitor_names[j], [rm_result.iloc[40 + i * 37][14 + j] / 10000 * 100 for i in range(27)]]
                      for j in range(competitor_len)]
    rm_query_time = [[competitor_names[j], [rm_result.iloc[42 + i * 37][14 + j] * 1000000 for i in range(27)]]
                     for j in range(competitor_len)]
    plot_T_lines(update_time_id_list, rm_retrain_avg_time, output_path + "/rm1.png",
                 '合并周期（${T}$）', '重训练平均时间（$\mathrm{s}$）',
                 competitor_colors, competitor_markers, [0.99, 0.125, 0.97, 0.15], True, 'upper left')
    plot_T_lines(update_time_id_list, rm_retrain_num, output_path + "/rm2.png",
                 '合并周期（${T}$）', '重训练次数（$\mathrm{×100}$）',
                 competitor_colors, competitor_markers, [0.99, 0.09, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, rm_retrain_time, output_path + "/rm3.png",
                 '合并周期（${T}$）', '重训练时间（$\mathrm{μs}$）',
                 competitor_colors, competitor_markers, [0.99, 0.14, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, rm_update_time, output_path + "/rm4.png",
                 '合并周期（${T}$）', '更新时间（$\mathrm{μs}$）',
                 competitor_colors, competitor_markers, [0.99, 0.14, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, rm_error_ratio, output_path + "/rm5.png",
                 '合并周期（${T}$）', '误差率（$\mathrm{\%}$）',
                 competitor_colors, competitor_markers, [0.99, 0.115, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, rm_query_time, output_path + "/rm6.png",
                 '合并周期（${T}$）', '检索时间（$\mathrm{μs}$）',
                 competitor_colors, competitor_markers, [0.99, 0.115, 0.97, 0.15], False, 'upper left')


def plot_compare_size(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    cp_result = pd.ExcelFile.parse(xls, sheet_name='compare', header=None)
    competitor_names = ['$\mathrm{RT}$', '$\mathrm{PRQT}$', '$\mathrm{BRINS}$', '$\mathrm{LISA}$',
                        '$\mathrm{TSUSLI}$', '$\mathrm{USLBRIN-NPS}$', '$\mathrm{USLBRIN}$']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#86B2C5',
                         '#FCE166', '#F53935', '#B71C1C']
    competitor_markers = ['v', 's', 'p', 'x', '*', 'o', 'o']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # size
    index_sizes = [[competitor_names[j],
                    [cp_result.iloc[4][71 + j + 7 * i] / 1024 / 1024 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    data_sizes = [[competitor_names[j],
                   [cp_result.iloc[5][71 + j + 7 * i] / 1024 / 1024 / 1024 for i in range(dataset_len)]]
                  for j in range(competitor_len)]
    sum_sizes = [[competitor_names[j],
                  [(cp_result.iloc[4][71 + j + 7 * i] + cp_result.iloc[5][71 + j + 7 * i]) / 1024 / 1024 / 1024
                   for i in range(dataset_len)]] for j in range(competitor_len)]
    update_time_id_list = list(range(1, 29, 1))
    update_index_sizes = [[competitor_names[j],
                           [cp_result.iloc[5 + i * 12 * 6][61 + j] / 1024 / 1024
                            for i in range(28)]] for j in range(competitor_len)]
    update_data_sizes = [[competitor_names[j],
                          [cp_result.iloc[6 + i * 12 * 6][61 + j] / 1024 / 1024 / 1024
                           for i in range(28)]] for j in range(competitor_len)]
    update_sum_sizes = [[competitor_names[j],
                         [(cp_result.iloc[5 + i * 12 * 6][61 + j] + cp_result.iloc[6 + i * 12][
                             61 + j]) / 1024 / 1024 / 1024
                          for i in range(28)]] for j in range(competitor_len)]
    # plot_group_histogram(datasets, data_sizes, output_path + "/size1.png",
    #                      '数据集', '数据体积（$\mathrm{GB}$）', competitor_colors,
    #                      [0.99, 0.12, 0.97, 0.15], [], False, None, False)
    plot_group_histogram(datasets, index_sizes, output_path + "/size2.png",
                         '数据集', '索引体积（$\mathrm{MB}$）', competitor_colors,
                         [0.99, 0.145, 0.97, 0.15], [], False, 'best', True)
    # plot_group_histogram(datasets, sum_sizes, output_path + "/size3.png",
    #                      '数据集', '存储成本（$\mathrm{GB}$）', competitor_colors,
    #                      [0.99, 0.12, 0.97, 0.15], [], False, None, False)
    # plot_T_lines(update_time_id_list, update_data_sizes, output_path + "/size4.png",
    #              '合并周期（${T}$）', '数据体积（$\mathrm{GB}$）', competitor_colors, competitor_markers,
    #              [0.99, 0.12, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, update_index_sizes, output_path + "/size5.png",
                 '合并周期（${T}$）', '索引体积（$\mathrm{MB}$）', competitor_colors, competitor_markers,
                 [0.99, 0.145, 0.97, 0.15], False, 'upper left', True)
    # plot_T_lines(update_time_id_list, update_sum_sizes, output_path + "/size6.png",
    #              '合并周期（${T}$）', '存储成本（$\mathrm{GB}$）', competitor_colors, competitor_markers,
    #              [0.99, 0.12, 0.97, 0.15], False, 'upper left')


def plot_compare_build(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    cp_result = pd.ExcelFile.parse(xls, sheet_name='compare', header=None)
    competitor_names = ['$\mathrm{RT}$', '$\mathrm{PRQT}$', '$\mathrm{BRINS}$', '$\mathrm{LISA}$',
                        '$\mathrm{TSUSLI}$', '$\mathrm{USLBRIN-NPS}$', '$\mathrm{USLBRIN}$']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#86B2C5',
                         '#FCE166', '#F53935', '#B71C1C']
    competitor_markers = ['v', 's', 'p', 'x', '*', 'o', 'o']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # build
    build_times = [[competitor_names[j],
                    [cp_result.iloc[12][71 + j + 7 * i] for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, build_times, output_path + "/build1.png",
                         '数据集', '构建时间（$\mathrm{s}$）', competitor_colors,
                         [0.99, 0.13, 0.97, 0.15], [], False, 'best', True)


def plot_compare_query(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    cp_result = pd.ExcelFile.parse(xls, sheet_name='compare', header=None)
    competitor_names = ['$\mathrm{RT}$', '$\mathrm{PRQT}$', '$\mathrm{BRINS}$', '$\mathrm{LISA}$',
                        '$\mathrm{TSUSLI}$', '$\mathrm{USLBRIN-NPS}$', '$\mathrm{USLBRIN}$']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#86B2C5',
                         '#FCE166', '#F53935', '#B71C1C']
    competitor_markers = ['v', 's', 'p', 'x', '*', 'o', 'o']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # query time
    point_query_times = [[competitor_names[j],
                          [cp_result.iloc[6][71 + j + 7 * i] * 1000000 for i in range(dataset_len)]]
                         for j in range(competitor_len)]
    point_query_ios = [[competitor_names[j],
                        [cp_result.iloc[7][71 + j + 7 * i] for i in range(dataset_len)]]
                       for j in range(competitor_len)]
    range_query_times = [[competitor_names[j],
                          [cp_result.iloc[8][71 + j + 7 * i] * 1000000 for i in range(dataset_len)]]
                         for j in range(competitor_len)]
    range_query_ios = [[competitor_names[j],
                        [cp_result.iloc[9][71 + j + 7 * i] for i in range(dataset_len)]]
                       for j in range(competitor_len)]
    knn_query_times = [[competitor_names[j],
                        [cp_result.iloc[10][71 + j + 7 * i] * 1000000 for i in range(dataset_len)]]
                       for j in range(competitor_len)]
    knn_query_ios = [[competitor_names[j],
                      [cp_result.iloc[11][71 + j + 7 * i] for i in range(dataset_len)]]
                     for j in range(competitor_len)]
    plot_group_histogram(datasets, point_query_ios, output_path + "/point_query1.png",
                         '数据集', '检索$\mathrm{IO}$', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    plot_group_histogram(datasets, point_query_times, output_path + "/point_query2.png",
                         '数据集', '检索时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    plot_group_histogram(datasets, range_query_ios, output_path + "/range_query1.png",
                         '数据集', '检索$\mathrm{IO}$', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    plot_group_histogram(datasets, range_query_times, output_path + "/range_query2.png",
                         '数据集', '检索时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    plot_group_histogram(datasets, knn_query_ios, output_path + "/knn_query1.png",
                         '数据集', '检索$\mathrm{IO}$', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    plot_group_histogram(datasets, knn_query_times, output_path + "/knn_query2.png",
                         '数据集', '检索时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    update_point_query_times = [[competitor_names[j],
                                 [cp_result.iloc[7 + i * 12 * 6][61 + j] * 1000000
                                  for i in range(28)]] for j in range(competitor_len)]
    update_point_query_ios = [[competitor_names[j],
                               [cp_result.iloc[8 + i * 12 * 6][61 + j]
                                for i in range(28)]] for j in range(competitor_len)]
    update_range_query_times = [[competitor_names[j],
                                 [cp_result.iloc[9 + i * 12 * 6][61 + j] * 1000000
                                  for i in range(28)]] for j in range(competitor_len)]
    update_range_query_ios = [[competitor_names[j],
                               [cp_result.iloc[10 + i * 12 * 6][61 + j]
                                for i in range(28)]] for j in range(competitor_len)]
    update_knn_query_times = [[competitor_names[j],
                               [cp_result.iloc[11 + i * 12 * 6][61 + j] * 1000000
                                for i in range(28)]] for j in range(competitor_len)]
    update_knn_query_ios = [[competitor_names[j],
                             [cp_result.iloc[12 + i * 12 * 6][61 + j]
                              for i in range(28)]] for j in range(competitor_len)]
    update_time_id_list = list(range(1, 29, 1))
    plot_T_lines(update_time_id_list, update_point_query_ios, output_path + "/point_query3.png",
                 '合并周期（${T}$）', '检索$\mathrm{IO}$', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_point_query_times, output_path + "/point_query4.png",
                 '合并周期（${T}$）', '检索时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_range_query_ios, output_path + "/range_query3.png",
                 '合并周期（${T}$）', '检索$\mathrm{IO}$', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_range_query_times, output_path + "/range_query4.png",
                 '合并周期（${T}$）', '检索时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_knn_query_ios, output_path + "/knn_query3.png",
                 '合并周期（${T}$）', '检索$\mathrm{IO}$', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_knn_query_times, output_path + "/knn_query4.png",
                 '合并周期（${T}$）', '检索时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)


def plot_compare_update(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    cp_result = pd.ExcelFile.parse(xls, sheet_name='compare', header=None)
    competitor_names = ['$\mathrm{RT}$', '$\mathrm{PRQT}$', '$\mathrm{BRINS}$', '$\mathrm{LISA}$',
                        '$\mathrm{TSUSLI}$', '$\mathrm{USLBRIN-NPS}$', '$\mathrm{USLBRIN}$']
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#86B2C5',
                         '#FCE166', '#F53935', '#B71C1C']
    competitor_markers = ['v', 's', 'p', 'x', '*', 'o', 'o']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # update time
    update_times = [[competitor_names[j],
                     [cp_result.iloc[2][71 + j + 7 * i] / cp_result.iloc[1][71 + j + 7 * i] * 1000000
                      for i in range(dataset_len)]]
                    for j in range(competitor_len)]
    plot_group_histogram(datasets, update_times, output_path + "/update1.png",
                         '数据集', '更新时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], False, 'best', True)
    update_uniform_times = [[competitor_names[j],
                             [(cp_result.iloc[3 + 12 * 0 + i * 12 * 6][47 + j] +
                               cp_result.iloc[3 + 12 * 1 + i * 12 * 6][47 + j] +
                               cp_result.iloc[3 + 12 * 2 + i * 12 * 6][47 + j] +
                               cp_result.iloc[3 + 12 * 3 + i * 12 * 6][47 + j] +
                               cp_result.iloc[3 + 12 * 4 + i * 12 * 6][47 + j] +
                               cp_result.iloc[3 + 12 * 5 + i * 12 * 6][47 + j]) / (
                                      cp_result.iloc[2 + 12 * 0 + i * 12 * 6][47 + j] +
                                      cp_result.iloc[2 + 12 * 1 + i * 12 * 6][47 + j] +
                                      cp_result.iloc[2 + 12 * 2 + i * 12 * 6][47 + j] +
                                      cp_result.iloc[2 + 12 * 3 + i * 12 * 6][47 + j] +
                                      cp_result.iloc[2 + 12 * 4 + i * 12 * 6][47 + j] +
                                      cp_result.iloc[2 + 12 * 5 + i * 12 * 6][47 + j]) * 1000000
                              for i in range(28)]] for j in range(competitor_len)]
    update_normal_times = [[competitor_names[j],
                            [(cp_result.iloc[3 + 12 * 0 + i * 12 * 6][54 + j] +
                              cp_result.iloc[3 + 12 * 1 + i * 12 * 6][54 + j] +
                              cp_result.iloc[3 + 12 * 2 + i * 12 * 6][54 + j] +
                              cp_result.iloc[3 + 12 * 3 + i * 12 * 6][54 + j] +
                              cp_result.iloc[3 + 12 * 4 + i * 12 * 6][54 + j] +
                              cp_result.iloc[3 + 12 * 5 + i * 12 * 6][54 + j]) / (
                                     cp_result.iloc[2 + 12 * 0 + i * 12 * 6][54 + j] +
                                     cp_result.iloc[2 + 12 * 1 + i * 12 * 6][54 + j] +
                                     cp_result.iloc[2 + 12 * 2 + i * 12 * 6][54 + j] +
                                     cp_result.iloc[2 + 12 * 3 + i * 12 * 6][54 + j] +
                                     cp_result.iloc[2 + 12 * 4 + i * 12 * 6][54 + j] +
                                     cp_result.iloc[2 + 12 * 5 + i * 12 * 6][54 + j]) * 1000000
                             for i in range(28)]] for j in range(competitor_len)]
    update_nyct_times = [[competitor_names[j],
                          [(cp_result.iloc[3 + 12 * 0 + i * 12 * 6][61 + j] +
                            cp_result.iloc[3 + 12 * 1 + i * 12 * 6][61 + j] +
                            cp_result.iloc[3 + 12 * 2 + i * 12 * 6][61 + j] +
                            cp_result.iloc[3 + 12 * 3 + i * 12 * 6][61 + j] +
                            cp_result.iloc[3 + 12 * 4 + i * 12 * 6][61 + j] +
                            cp_result.iloc[3 + 12 * 5 + i * 12 * 6][61 + j]) / (
                                   cp_result.iloc[2 + 12 * 0 + i * 12 * 6][61 + j] +
                                   cp_result.iloc[2 + 12 * 1 + i * 12 * 6][61 + j] +
                                   cp_result.iloc[2 + 12 * 2 + i * 12 * 6][61 + j] +
                                   cp_result.iloc[2 + 12 * 3 + i * 12 * 6][61 + j] +
                                   cp_result.iloc[2 + 12 * 4 + i * 12 * 6][61 + j] +
                                   cp_result.iloc[2 + 12 * 5 + i * 12 * 6][61 + j]) * 1000000
                           for i in range(28)]] for j in range(competitor_len)]
    update_time_id_list = list(range(1, 29, 1))
    plot_T_lines(update_time_id_list, update_uniform_times, output_path + "/update2.png",
                 '合并周期（${T}$）', '更新时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_normal_times, output_path + "/update3.png",
                 '合并周期（${T}$）', '更新时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    plot_T_lines(update_time_id_list, update_nyct_times, output_path + "/update4.png",
                 '合并周期（${T}$）', '更新时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)
    update_nyct_nums = [[None,
                         [(cp_result.iloc[2 + 12 * 0 + i * 12 * 6][61] +
                           cp_result.iloc[2 + 12 * 1 + i * 12 * 6][61] +
                           cp_result.iloc[2 + 12 * 2 + i * 12 * 6][61] +
                           cp_result.iloc[2 + 12 * 3 + i * 12 * 6][61] +
                           cp_result.iloc[2 + 12 * 4 + i * 12 * 6][61] +
                           cp_result.iloc[2 + 12 * 5 + i * 12 * 6][61]) / 100000
                          for i in range(28)]]]
    plot_T_lines(update_time_id_list, update_nyct_nums, output_path + "/update5.png",
                 '合并周期（${T}$）', '更新数据体量（$\mathrm{×100000}$）', None, None,
                 [0.99, 0.125, 0.97, 0.15], False, 'upper left', False)
    update_nyct_sum_times = [[competitor_names[j],
                              [(cp_result.iloc[3 + 12 * 0 + i * 12 * 6][61 + j] +
                                cp_result.iloc[3 + 12 * 1 + i * 12 * 6][61 + j] +
                                cp_result.iloc[3 + 12 * 2 + i * 12 * 6][61 + j] +
                                cp_result.iloc[3 + 12 * 3 + i * 12 * 6][61 + j] +
                                cp_result.iloc[3 + 12 * 4 + i * 12 * 6][61 + j] +
                                cp_result.iloc[3 + 12 * 5 + i * 12 * 6][61 + j])
                               for i in range(28)]] for j in range(competitor_len)]
    plot_T_lines(update_time_id_list, update_nyct_sum_times, output_path + "/update6.png",
                 '合并周期（${T}$）', '更新总时间（$\mathrm{s}$）', competitor_colors, competitor_markers,
                 [0.99, 0.135, 0.97, 0.15], False, 'upper left', True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    input_path = "./table/result_uslbrin.xlsx"
    output_path = "./result_uslbrin"
    plot_grid_search_tel(input_path, output_path)
    plot_grid_search_tef(input_path, output_path)
    plot_grid_search_ten(input_path, output_path)
    plot_grid_search_rm(input_path, output_path)
    plot_compare_size(input_path, output_path)
    plot_compare_build(input_path, output_path)
    plot_compare_query(input_path, output_path)
    plot_compare_update(input_path, output_path)
