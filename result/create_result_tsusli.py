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
                 legend_pos='best'):
    marker_size = 10
    major_locator = 6
    minor_locator = 1
    ax = plt.axes()
    ax.xaxis.set_major_locator(MultipleLocator(major_locator))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_locator))
    for i in range(len(y_list)):
        # 平均值平滑
        # for j in range(1, len(y_list[i][1]) - 1):
        #     y_list[i][1][j] = (y_list[i][1][j + 1] + y_list[i][1][j - 1]) / 2
        plt.plot(x, y_list[i][1], label=y_list[i][0],
                 color=color_list[i], marker=marker_list[i], markersize=marker_size)
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlim(0, 24)
    # plt.yscale("log")
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
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    plt.gcf().subplots_adjust(right=adjust[0], left=adjust[1], top=adjust[2], bottom=adjust[3])
    if is_legend:
        plt.legend(loc=legend_pos, frameon=False, ncol=1, columnspacing=1, handletextpad=0.3, fontsize=font_size)
    plt.savefig(model_path)
    plt.close()


def plot_grid_search_f(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    f_result = pd.ExcelFile.parse(xls, sheet_name='f', header=None)
    f_x = ['1', '6', '12', '18', '24']
    f_pre_max_key_mae = [f_result.iloc[38][16 + i] for i in range(5)]
    f_pre_cdf_mae = [f_result.iloc[36][16 + i] * 100 for i in range(5)]
    f_true_max_key_mae = [f_result.iloc[37][16 + i] for i in range(5)]
    f_true_cdf_mae = [f_result.iloc[35][16 + i] * 100 for i in range(5)]
    f_retrain_num = [f_result.iloc[39][16 + i] for i in range(5)]
    f_retrain_avg_time = [f_result.iloc[1][16 + i] for i in range(5)]
    f_update_time = [f_result.iloc[40][16 + i] * 1000 for i in range(5)]
    f_query_time = [f_result.iloc[17][16 + i] * 1000000 for i in range(5)]
    f_x_title = "$\mathit{f}$"
    plot_histogram_and_line(f_x, f_pre_max_key_mae, f_pre_cdf_mae, output_path + "/f1.png", f_x_title,
                            "空间密度预测误差", "空间排列预测误差（$\mathrm{\%}$）",
                            "空间密度", "空间排列",
                            color1, color2, [0.89, 0.09, 0.97, 0.15], [0, 5, 1], [10, 20, 2], True, (1, 1))
    plot_histogram_and_line(f_x, f_true_max_key_mae, f_true_cdf_mae, output_path + "/f2.png", f_x_title,
                            "空间密度真实误差", "空间排列真实误差（$\mathrm{\%}$）",
                            "空间密度", "空间排列",
                            color1, color2, [0.88, 0.11, 0.97, 0.15], [2, 12, 2], [2.8, 3.8, 0.2], True, (1, 1))
    plot_histogram_and_line(f_x, f_retrain_num, f_retrain_avg_time, output_path + "/f3.png", f_x_title,
                            "重训练次数", "重训练平均时间（$\mathrm{s}$）",
                            "重训练次数", "重训练时间",
                            color1, color2, [0.91, 0.13, 0.97, 0.15], [5, 1000, 100], [0, 5, 1], True, (1, 1), True)
    plot_line_and_line(f_x, f_update_time, f_query_time, output_path + "/f4.png", f_x_title,
                       "更新时间（$\mathrm{ms}$）", "检索时间（$\mathrm{μs}$）",
                       "更新时间", "检索时间",
                       color1, color2, [0.85, 0.115, 0.97, 0.15], [6, 10, 1], [30.8, 31.6, 0.2], True, (1, 1))


def plot_grid_search_c(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    c_result = pd.ExcelFile.parse(xls, sheet_name='c', header=None)
    c_x = ['10', '50', '100', '250', '500']
    c_pre_max_key_mae = [c_result.iloc[38][16 + i] for i in range(5)]
    c_pre_cdc_mae = [c_result.iloc[36][16 + i] * 100 for i in range(5)]
    c_true_max_key_mae = [c_result.iloc[37][16 + i] for i in range(5)]
    c_true_cdc_mae = [c_result.iloc[35][16 + i] * 100 for i in range(5)]
    c_data_size = [c_result.iloc[41][16 + i] / 1024 / 1024 for i in range(5)]
    c_retrain_avg_time = [c_result.iloc[1][16 + i] for i in range(5)]
    c_update_time = [c_result.iloc[40][16 + i] * 1000 for i in range(5)]
    c_query_time = [c_result.iloc[17][16 + i] * 1000000 for i in range(5)]
    c_x_title = "$\mathit{c}$"
    plot_histogram_and_line(c_x, c_pre_max_key_mae, c_pre_cdc_mae, output_path + "/c1.png", c_x_title,
                            "空间密度预测误差", "空间排列预测误差（$\mathrm{\%}$）",
                            "空间密度", "空间排列",
                            color1, color2, [0.89, 0.09, 0.97, 0.15], [1, 6, 1], [14, 19, 1], True, (1, 1))
    plot_histogram_and_line(c_x, c_true_max_key_mae, c_true_cdc_mae, output_path + "/c2.png", c_x_title,
                            "空间密度真实误差", "空间排列真实误差（$\mathrm{\%}$）",
                            "空间密度", "空间排列",
                            color1, color2, [0.88, 0.11, 0.97, 0.15], [4, 12, 2], [3.1, 3.5, 0.1], True, (1, 1))
    plot_histogram_and_line(c_x, c_data_size, c_retrain_avg_time, output_path + "/c3.png", c_x_title,
                            "索引体积（$\mathrm{MB}$）", "重训练平均时间（$\mathrm{s}$）",
                            "索引体积", "训练时间",
                            color1, color2, [0.91, 0.11, 0.97, 0.15], [], [0, 4, 1], True, (0.85, 1))
    plot_line_and_line(c_x, c_update_time, c_query_time, output_path + "/c4.png", c_x_title,
                       "更新时间（$\mathrm{ms}$）", "检索时间（$\mathrm{μs}$）",
                       "更新时间", "检索时间",
                       color1, color2, [0.85, 0.115, 0.97, 0.15], [5, 10, 1], [31, 31.8, 0.2], True, (1, 1))


def plot_grid_search_bs(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    color1 = '#808080'
    color2 = '#B71C1C'
    bs_result = pd.ExcelFile.parse(xls, sheet_name='bs', header=None)
    bs_x = ['1', '100', '200', '300', '400']
    bs_data_size = [bs_result.iloc[33][16 + i] / 1024 / 1024 / 1024 for i in range(5)]
    bs_insert_time = [bs_result.iloc[25][16 + i] * 1000000 for i in range(5)]
    bs_update_time = [bs_result.iloc[40][16 + i] * 1000 for i in range(5)]
    bs_query_time = [bs_result.iloc[17][16 + i] * 1000000 for i in range(5)]
    bs_x_title = "$\mathit{bs}$"
    plot_histogram_and_line(bs_x, bs_data_size, bs_insert_time, output_path + "/bs1.png", bs_x_title,
                            "数据体积（$\mathrm{GB}$）", "定位时间（$\mathrm{μs}$）",
                            "数据体积", "定位时间",
                            color1, color2, [0.85, 0.09, 0.97, 0.15], [0, 4, 1], [16.5, 18.5, 0.5], True, (1, 1))
    plot_line_and_line(bs_x, bs_update_time, bs_query_time, output_path + "/bs2.png", bs_x_title,
                       "更新时间（$\mathrm{ms}$）", "检索时间（$\mathrm{μs}$）",
                       "更新时间", "检索时间",
                       color1, color2, [0.85, 0.115, 0.97, 0.15], [6, 10, 1], [31.2, 32, 0.2], True, (1, 1))


def plot_grid_search_sts(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    ts_result = pd.ExcelFile.parse(xls, sheet_name='MF', header=None)
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5']
    competitor_markers = ['v', 's', 'p']
    competitor_names = ['$\mathrm{VSARIMA}$', '$\mathrm{FCLSTM}$', '$\mathrm{CONVLSTM}$']
    competitor_len = len(competitor_names)
    update_time_id_list = list(range(1, 25, 1))
    ts_pre_max_key_mae = [[competitor_names[j], [ts_result.iloc[1030 + i * 32][16 + j] * 100 for i in range(24)]]
                          for j in range(competitor_len)]
    ts_true_max_key_mae = [[competitor_names[j], [ts_result.iloc[1025 + i * 32][16 + j] * 100 for i in range(24)]]
                           for j in range(competitor_len)]
    ts_retrain_avg_time = [[competitor_names[j], [ts_result.iloc[1032 + i * 32][16 + j] /
                                                  ts_result.iloc[1029 + i * 32][16 + j]
                                                  if ts_result.iloc[1029 + i * 32][16 + j] else 0
                                                  for i in range(24)]]
                           for j in range(competitor_len)]
    ts_update_time = [[competitor_names[j], [(ts_result.iloc[1023 + i * 32][16 + j] +
                                              ts_result.iloc[1027 + i * 32][16 + j] +
                                              ts_result.iloc[1032 + i * 32][16 + j] +
                                              126.0583806) /
                                             (ts_result.iloc[1007 + i * 32][16 + j] +
                                              ts_result.iloc[1010 + i * 32][16 + j] +
                                              ts_result.iloc[1013 + i * 32][16 + j] +
                                              ts_result.iloc[1016 + i * 32][16 + j] +
                                              ts_result.iloc[1019 + i * 32][16 + j] +
                                              ts_result.iloc[1036 + i * 32][16 + j]) * 10
                                             for i in range(24)]]
                      for j in range(competitor_len)]
    ts_query_time = [[competitor_names[j], [(ts_result.iloc[1008 + i * 32][16 + j] +
                                             ts_result.iloc[1011 + i * 32][16 + j] +
                                             ts_result.iloc[1014 + i * 32][16 + j] +
                                             ts_result.iloc[1017 + i * 32][16 + j] +
                                             ts_result.iloc[1020 + i * 32][16 + j] +
                                             ts_result.iloc[1037 + i * 32][16 + j]) / 6 * 1000000
                                            for i in range(24)]]
                     for j in range(competitor_len)]
    plot_T_lines(update_time_id_list, ts_pre_max_key_mae, output_path + "/sts1.png",
                 '合并周期（${T}$）', '空间排列预测误差（$\mathrm{\%}$）',
                 competitor_colors, competitor_markers, [0.975, 0.115, 0.97, 0.15], False, 'center left')
    plot_T_lines(update_time_id_list, ts_true_max_key_mae, output_path + "/sts2.png",
                 '合并周期（${T}$）', '空间排列真实误差（$\mathrm{\%}$）',
                 competitor_colors, competitor_markers, [0.975, 0.125, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, ts_retrain_avg_time, output_path + "/sts3.png",
                 '合并周期（${T}$）', '重训练平均时间（$\mathrm{s}$）',
                 competitor_colors, competitor_markers, [0.975, 0.135, 0.97, 0.15], True, 'upper left')
    plot_T_lines(update_time_id_list, ts_update_time, output_path + "/sts4.png",
                 '合并周期（${T}$）', '更新时间（$\mathrm{100ms}$）',
                 competitor_colors, competitor_markers, [0.975, 0.115, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, ts_query_time, output_path + "/sts5.png",
                 '合并周期（${T}$）', '检索时间（$\mathrm{μs}$）',
                 competitor_colors, competitor_markers, [0.975, 0.115, 0.97, 0.15], False, 'upper left')


def plot_grid_search_ts(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    ts_result = pd.ExcelFile.parse(xls, sheet_name='Mn', header=None)
    competitor_colors = ['#95CCBA', '#F2C477', '#BFC0D5', '#FCE166', '#86B2C5']
    competitor_markers = ['v', 's', 'p', '*', 'x']
    competitor_names = ['$\mathrm{ES}$', '$\mathrm{SARIMA}$', '$\mathrm{RNN}$', '$\mathrm{LSTM}$', '$\mathrm{GRU}$']
    competitor_len = len(competitor_names)
    update_time_id_list = list(range(1, 25, 1))
    ts_pre_max_key_mae = [[competitor_names[j], [ts_result.iloc[1031 + i * 32][16 + j] for i in range(24)]]
                          for j in range(competitor_len)]
    ts_true_max_key_mae = [[competitor_names[j], [ts_result.iloc[1026 + i * 32][16 + j] for i in range(24)]]
                           for j in range(competitor_len)]
    ts_retrain_avg_time = [[competitor_names[j], [ts_result.iloc[1032 + i * 32][16 + j] /
                                                  ts_result.iloc[1029 + i * 32][16 + j]
                                                  if ts_result.iloc[1029 + i * 32][16 + j] else 0
                                                  for i in range(24)]]
                           for j in range(competitor_len)]
    ts_update_time = [[competitor_names[j], [(ts_result.iloc[1023 + i * 32][16 + j] +
                                              ts_result.iloc[1027 + i * 32][16 + j] +
                                              ts_result.iloc[1032 + i * 32][16 + j] +
                                              126.0583806) /
                                             (ts_result.iloc[1007 + i * 32][16 + j] +
                                              ts_result.iloc[1010 + i * 32][16 + j] +
                                              ts_result.iloc[1013 + i * 32][16 + j] +
                                              ts_result.iloc[1016 + i * 32][16 + j] +
                                              ts_result.iloc[1019 + i * 32][16 + j] +
                                              ts_result.iloc[1036 + i * 32][16 + j]) * 10
                                             for i in range(24)]]
                      for j in range(competitor_len)]
    ts_query_time = [[competitor_names[j], [(ts_result.iloc[1008 + i * 32][16 + j] +
                                             ts_result.iloc[1011 + i * 32][16 + j] +
                                             ts_result.iloc[1014 + i * 32][16 + j] +
                                             ts_result.iloc[1017 + i * 32][16 + j] +
                                             ts_result.iloc[1020 + i * 32][16 + j] +
                                             ts_result.iloc[1037 + i * 32][16 + j]) / 6 * 1000000
                                            for i in range(24)]]
                     for j in range(competitor_len)]
    plot_T_lines(update_time_id_list, ts_pre_max_key_mae, output_path + "/ts1.png",
                 '合并周期（${T}$）', '空间密度预测误差',
                 competitor_colors, competitor_markers, [0.975, 0.12, 0.97, 0.15], False, 'center left')
    plot_T_lines(update_time_id_list, ts_true_max_key_mae, output_path + "/ts2.png",
                 '合并周期（${T}$）', '空间密度真实误差',
                 competitor_colors, competitor_markers, [0.975, 0.11, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, ts_retrain_avg_time, output_path + "/ts3.png",
                 '合并周期（${T}$）', '重训练平均时间（$\mathrm{s}$）',
                 competitor_colors, competitor_markers, [0.975, 0.115, 0.97, 0.15], True, 'upper left')
    plot_T_lines(update_time_id_list, ts_update_time, output_path + "/ts4.png",
                 '合并周期（${T}$）', '更新时间（$\mathrm{100ms}$）',
                 competitor_colors, competitor_markers, [0.975, 0.095, 0.97, 0.15], False, 'upper left')
    plot_T_lines(update_time_id_list, ts_query_time, output_path + "/ts5.png",
                 '合并周期（${T}$）', '检索时间（$\mathrm{μs}$）',
                 competitor_colors, competitor_markers, [0.975, 0.115, 0.97, 0.15], False, 'upper left')


def plot_compare_size_query_update(input_path, output_path):
    xls = pd.ExcelFile(input_path)
    cp_result = pd.ExcelFile.parse(xls, sheet_name='compare', header=None)
    competitor_names = ['$\mathrm{IPUSLI-1}$', '$\mathrm{IPUSLI-2}$', '$\mathrm{DTUSLI}$', '$\mathrm{TSUSLI}$']
    competitor_colors = ['#F2C477', '#FCE166', '#95CCBA', '#B71C1C']
    competitor_markers = ['*', '*', 'v', 'o']
    datasets = ['UNIFORM', 'NORMAL', 'NYCT']
    competitor_len = len(competitor_names)
    dataset_len = len(datasets)
    # size
    data_sizes = [[competitor_names[j],
                   [cp_result.iloc[33][30 + j + 4 * i] / 1024 / 1024 / 1024 for i in range(dataset_len)]]
                  for j in range(competitor_len)]
    index_sizes = [[competitor_names[j],
                    [cp_result.iloc[41][30 + j + 4 * i] / 1024 / 1024 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    sum_sizes = [[competitor_names[j],
                  [(cp_result.iloc[33][30 + j + 4 * i] + cp_result.iloc[41][30 + j + 4 * i]) / 1024 / 1024 / 1024
                   for i in range(dataset_len)]] for j in range(competitor_len)]
    update_time_id_list = list(range(1, 25, 1))
    update_sum_sizes = [[competitor_names[j],
                         [(cp_result.iloc[41][38 + j] + cp_result.iloc[25 + i * 25][52 + j]) / 1024 / 1024 / 1024
                          for i in range(24)]] for j in range(competitor_len)]
    plot_group_histogram(datasets, data_sizes, output_path + "/size1.png",
                         '数据集', '数据体积（$\mathrm{GB}$）', competitor_colors,
                         [0.99, 0.12, 0.97, 0.15], [1.2, 2, 0.2], True, 'upper left', False)
    plot_group_histogram(datasets, index_sizes, output_path + "/size2.png",
                         '数据集', '索引体积（$\mathrm{MB}$）', competitor_colors,
                         [0.99, 0.11, 0.97, 0.15], [0, 18, 3], False, None, False)
    plot_group_histogram(datasets, sum_sizes, output_path + "/size3.png",
                         '数据集', '存储成本（$\mathrm{GB}$）', competitor_colors,
                         [0.99, 0.12, 0.97, 0.15], [1.2, 2, 0.2], False, None, False)
    plot_T_lines(update_time_id_list, update_sum_sizes, output_path + "/size4.png",
                 '合并周期（${T}$）', '存储成本（$\mathrm{GB}$）', competitor_colors, competitor_markers,
                 [0.975, 0.12, 0.97, 0.15], True, 'center left')
    # query time
    err_ranges = [[competitor_names[j], [cp_result.iloc[34][30 + j + 4 * i] / 1000 for i in range(dataset_len)]]
                  for j in range(competitor_len)]
    plot_group_histogram(datasets, err_ranges, output_path + "/query1.png",
                         '数据集', '误差范围（$\mathrm{×1000}$）', competitor_colors,
                         [0.99, 0.09, 0.97, 0.15], [0, 8, 2], True, 'upper left', False)
    update_time_id_list = list(range(1, 25, 1))
    update_err_ranges = [[competitor_names[j], [cp_result.iloc[26 + i * 25][52 + j] / 1000 for i in range(24)]]
                         for j in range(competitor_len)]
    plot_T_lines(update_time_id_list, update_err_ranges, output_path + "/query2.png",
                 '合并周期（${T}$）', '误差范围（$\mathrm{×1000}$）',
                 competitor_colors, competitor_markers, [0.975, 0.09, 0.97, 0.15], True, 'center left')

    query_times = [[competitor_names[j], [cp_result.iloc[17][30 + j + 4 * i] * 1000000 for i in range(dataset_len)]]
                   for j in range(competitor_len)]
    plot_group_histogram(datasets, query_times, output_path + "/query3.png",
                         '数据集', '检索时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.12, 0.97, 0.15], [24, 32, 2], False, None, False)
    id_list = np.arange(1, 13) / 6
    cols = [6, 9, 12, 15, 18, 28, 31, 34, 37, 40, 43, 53]
    udpate_query_times = [[competitor_names[j], [cp_result.iloc[col][52 + j] * 1000000 for col in cols]]
                          for j in range(competitor_len)]
    plot_lines(id_list, udpate_query_times, output_path + "/query4.png",
               "合并周期（${T}$）", '检索时间（$\mathrm{μs}$）', competitor_colors, competitor_markers,
               [0.975, 0.12, 0.97, 0.15], [0, 2, 1, 1 / 6], [28, 32.5, 1], False, 'upper left', False)
    # update time
    insert_times = [[competitor_names[j], [(cp_result.iloc[25][30 + j + 4 * i]) * 1000000 for i in range(dataset_len)]]
                    for j in range(competitor_len)]
    adjust_times = [[competitor_names[j], [(cp_result.iloc[27][30 + j + 4 * i]) * 1000000 for i in range(dataset_len)]]
                    for j in range(competitor_len)]
    retrain_times = [[competitor_names[j], [(cp_result.iloc[29][30 + j + 4 * i]
                                             + cp_result.iloc[31][30 + j + 4 * i]) * 1000 for i in range(dataset_len)]]
                     for j in range(competitor_len)]
    update_times = [[competitor_names[j], [(cp_result.iloc[40][30 + j + 4 * i]) * 1000 for i in range(dataset_len)]]
                    for j in range(competitor_len)]
    plot_group_histogram(datasets, insert_times, output_path + "/update1.png",
                         '数据集', '定位时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.135, 0.97, 0.15], [], True, 'upper left', True)
    plot_group_histogram(datasets, adjust_times, output_path + "/update2.png",
                         '数据集', '调整时间（$\mathrm{μs}$）', competitor_colors,
                         [0.99, 0.12, 0.97, 0.15], [0, 60, 10], False, None, False)
    plot_group_histogram(datasets, retrain_times, output_path + "/update3.png",
                         '数据集', '重训练时间（$\mathrm{ms}$）', competitor_colors,
                         [0.99, 0.115, 0.97, 0.15], [0, 20, 5], False, None, False)
    plot_group_histogram(datasets, update_times, output_path + "/update4.png",
                         '数据集', '更新时间（$\mathrm{ms}$）', competitor_colors,
                         [0.99, 0.115, 0.97, 0.15], [0, 30, 5], False, None, False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    input_path = "./table/result_tsusli.xlsx"
    output_path = "./result_tsusli"
    plot_grid_search_f(input_path, output_path)
    plot_grid_search_bs(input_path, output_path)
    plot_grid_search_c(input_path, output_path)
    plot_grid_search_sts(input_path, output_path)
    plot_grid_search_ts(input_path, output_path)
    plot_compare_size_query_update(input_path, output_path)
