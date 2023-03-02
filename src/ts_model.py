import multiprocessing
import os
import time
import warnings

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, ConvLSTM1D, Conv1D, Dropout, SimpleRNN, GRU
from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
from src.spatial_index.common_utils import binary_search_less_max_duplicate


class TimeSeriesModel:
    def __init__(self, model_path,
                 old_cdfs, cur_cdf, type_cdf,
                 old_max_keys, cur_max_key, type_max_key,
                 key_list):
        # common
        self.name = "Time Series Model"
        self.model_path = model_path
        # for ts of cdf
        self.old_cdfs = old_cdfs
        self.cur_cdf = cur_cdf
        self.model_cdf = type_model_dict[type_cdf]
        self.mse_cdf = 0
        # for ts of max key
        self.old_max_keys = old_max_keys
        self.cur_max_key = cur_max_key
        self.model_max_key = type_model_dict[type_max_key]
        self.mse_max_key = 0
        # for compute
        self.key_list = key_list

    def build(self, cdf_width, lag):
        old_cdfs_num = len(self.old_cdfs)
        if old_cdfs_num == 0:  # if old_cdfs is []
            self.cur_cdf = [0.0] * cdf_width
            self.cur_max_key = 0
        elif old_cdfs_num <= lag:  # if old_cdfs are not enough for ts_model in quantity
            self.cur_cdf = self.old_cdfs[-1]
            self.cur_max_key = self.old_max_keys[-1]
        else:
            ts = self.model_cdf(self.old_cdfs, lag, cdf_width, self.model_path)
            ts.grid_search(thread=4, start_num=1)
            self.cur_cdf, self.mse_cdf = ts.train()
            ts = self.model_max_key(self.old_max_keys, lag, self.model_path)
            # ts.grid_search(thread=3, start_num=0)
            self.cur_max_key, self.mse_max_key = ts.train()

    def update(self, data, cdf_width, lag):
        self.old_cdfs.append(build_cdf(data, cdf_width, self.key_list))
        self.old_max_keys.append(len(data) - 1)
        self.build(cdf_width, lag)


class TSResult:
    def __init__(self):
        self.model_path = None

    def print_grid_search_result(self):
        files = os.listdir(self.model_path)
        results = [file.rsplit("_", 2) for file in files if file.endswith('.png')]
        for result in results:
            result[-1] = float(result[-1].replace(".png", ""))
        results.sort(key=lambda x: x[-1])
        output_file = open(self.model_path + 'result.txt', 'w')
        template = '%s, %s, %s\n'
        for result in results:
            output_file.write(template % (result[2], result[1], result[0]))
        output_file.close()


class ESResult(TSResult):
    """
    Holt-Winters/三次指数平滑法：趋势 + 季节性， 季节周期=lag
    1. Brown/一次指数平滑法：无趋势或季节性
    2. Holt/二次指数平滑法：有趋势，无季节性
    3. Holt-Winters/三次指数平滑法(趋势方法，季节方法，季节周期=lag)：有趋势和季节性
    grid search:
    1. 趋势方法：[加法, 乘法]
    2. 季节方法：[加法, 乘法]
    加法对应线性: add/additive的结果相同
    乘法对应指数: mul/multiplicative的结果相同
    """

    def __init__(self, data, lag, model_path):
        super().__init__()
        self.data = data
        self.lag = lag
        self.model_path = model_path + 'es/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, trend, seasonal, is_plot=False):
        model = ExponentialSmoothing(self.data, seasonal_periods=self.lag, trend=trend, seasonal=seasonal).fit()
        pre = correct_max_key(model.forecast(steps=1))
        mse = model.sse / model.model.nobs
        if is_plot:
            plt.plot(self.data)
            plt.plot(model.predict(0, len(self.data) - 1))
            plt.savefig(
                self.model_path + "%s_%s_%s.png" % (trend, seasonal, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(trend='add', seasonal='add')

    def grid_search(self, thread=1, start_num=0):
        trends = ["add", "mul", "additive", "multiplicative", None]
        seasonals = ["add", "mul", "additive", "multiplicative", None]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for trend in trends:
            for seasonal in seasonals:
                i += 1
                if i > start_num:
                    pool.apply_async(self.build, (trend, seasonal, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class SARIMAResult(TSResult):
    """
    季节性ARIMA
    1. ARMA(p, q)：无趋势或季节性
    2. ARIMA(p, d, q)：有趋势，无季节性
    3. SARIMA(p, d, q, P, D, Q, S=lag)：有趋势和季节性
    grid search:
    1. 趋势：pdq
    2. 季节：PQD
    """

    def __init__(self, data, lag, model_path):
        super().__init__()
        self.data = np.array(data)
        self.lag = lag
        self.model_path = model_path + 'sarima/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, p, d, q, P, Q, D, is_plot=False):
        model = SARIMAX(self.data, order=(p, d, q), seasonal_order=(P, D, Q, self.lag),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        pre = correct_max_key(model.forecast(steps=1))
        mse = model.mse
        # sum([data ** 2 for data in model.predict(0, self.data.size - 1) - self.data]) / self.data.size
        if is_plot:
            plt.plot(self.data)
            plt.plot(model.predict(0, self.data.size - 1))
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s.png" % (p, d, q, P, D, Q, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(p=3, d=1, q=0, P=2, D=0, Q=3)

    def grid_search(self, thread=1, start_num=0):
        # from statsmodels.tsa.stattools import acf, pacf
        # acf_list = acf(self.data, nlags=10)
        # pacf_list = pacf(self.data, nlags=10)
        #       AR(p)         MA(q)            ARMA(p,q)
        # ACF   拖尾           截尾+q阶后为0     拖尾+q阶后为0
        # PACF  截尾+p阶后为0   拖尾             拖尾+p阶后为0
        # plot_acf(self.data, lags=10).show()
        # plot_pacf(self.data, lags=10).show()
        ps = ds = qs = [0, 1, 2, 3]
        Ps = Ds = Qs = [0, 1, 2, 3]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for p in ps:
            for d in ds:
                for q in qs:
                    for P in Ps:
                        for D in Ds:
                            for Q in Qs:
                                i += 1
                                if i > start_num:
                                    pool.apply_async(self.build, (p, d, q, P, D, Q, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class RNNResult(TSResult):
    """
    RNN
    grid search:
    1. 模型结构：激活函数、第一层神经元数量、第一层dropout比例、第二层神经元数量、第二层dropout比例
    2. 训练参数：学习率、批大小
    """

    def __init__(self, data, lag, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        data = np.array(data)
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i - lag:i] for i in range(lag, len(train_data))])
        self.train_y = np.array([train_data[i] for i in range(lag, len(train_data))])
        self.test_x = np.array([test_data[i - lag:i] for i in range(lag, len(test_data) + 1)])
        self.test_y = np.array([test_data[i] for i in range(lag, len(test_data))])
        self.lag = lag
        self.model_path = model_path + 'rnn/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        model = Sequential([
            SimpleRNN(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            SimpleRNN(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=1)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x[:-1], self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(history.model.predict(self.test_x[-1:])[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum([(pre - true) ** 2 for pre, true in zip(model.predict(self.test_x[:-1]), self.test_y)])[0] / self.test_y.size
        mse = history.history['val_loss'][-1]
        if is_plot:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0,
                          learning_rate=0.01, batch_size=4)

    def grid_search(self, thread=1, start_num=0):
        # TODO
        # activations = ['relu', 'leaky_relu', 'tanh']
        activations = ['relu']
        # unit1s = [32, 64, 128, 256, 516]
        unit1s = [64, 128, 256, 516]
        # unit1s = [32, 64, 128, 256, 516]
        unit2s = [64, 128, 256, 516]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0, 0.2]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.1, 0.01]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [2, 4, 8, 16]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for activation in activations:
            for unit1 in unit1s:
                for unit2 in unit2s:
                    for dropout1 in dropout1s:
                        for dropout2 in dropout2s:
                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    i += 1
                                    if i > start_num:
                                        pool.apply_async(self.build,
                                                         (activation, unit1, unit2, dropout1, dropout2, learning_rate,
                                                          batch_size, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class LSTMResult(TSResult):
    """
    LSTM
    grid search:
    1. 模型结构：激活函数、第一层神经元数量、第一层dropout比例、第二层神经元数量、第二层dropout比例
    2. 训练参数：学习率、批大小
    """

    def __init__(self, data, lag, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        data = np.array(data)
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i - lag:i] for i in range(lag, len(train_data))])
        self.train_y = np.array([train_data[i] for i in range(lag, len(train_data))])
        self.test_x = np.array([test_data[i - lag:i] for i in range(lag, len(test_data) + 1)])
        self.test_y = np.array([test_data[i] for i in range(lag, len(test_data))])
        self.lag = lag
        self.model_path = model_path + 'lstm/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        model = Sequential([
            LSTM(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            LSTM(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=1)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x[:-1], self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(history.model.predict(self.test_x[-1:])[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum([(pre - true) ** 2 for pre, true in zip(model.predict(self.test_x[:-1]), self.test_y)])[0] / self.test_y.size
        mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0,
                          learning_rate=0.01, batch_size=4)

    def grid_search(self, thread=1, start_num=0):
        # activations = ['relu', 'leaky_relu', 'tanh']
        activations = ['relu']
        # unit1s = [32, 64, 128, 256, 516]
        unit1s = [64, 128, 256, 516]
        # unit1s = [32, 64, 128, 256, 516]
        unit2s = [64, 128, 256, 516]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0, 0.2]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.1, 0.01]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [2, 4, 8, 16]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for activation in activations:
            for unit1 in unit1s:
                for unit2 in unit2s:
                    for dropout1 in dropout1s:
                        for dropout2 in dropout2s:
                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    i += 1
                                    if i > start_num:
                                        pool.apply_async(self.build,
                                                         (activation, unit1, unit2, dropout1, dropout2, learning_rate,
                                                          batch_size, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class GRUResult(TSResult):
    """
    GRU
    grid search:
    1. 模型结构：激活函数、第一层神经元数量、第一层dropout比例、第二层神经元数量、第二层dropout比例
    2. 训练参数：学习率、批大小
    """

    def __init__(self, data, lag, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        data = np.array(data)
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i - lag:i] for i in range(lag, len(train_data))])
        self.train_y = np.array([train_data[i] for i in range(lag, len(train_data))])
        self.test_x = np.array([test_data[i - lag:i] for i in range(lag, len(test_data) + 1)])
        self.test_y = np.array([test_data[i] for i in range(lag, len(test_data))])
        self.lag = lag
        self.model_path = model_path + 'gru/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        model = Sequential([
            GRU(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            GRU(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=1)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x[:-1], self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(history.model.predict(self.test_x[-1:])[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum([(pre - true) ** 2 for pre, true in zip(model.predict(self.test_x[:-1]), self.test_y)])[0] / self.test_y.size
        mse = history.history['val_loss'][-1]
        if is_plot:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0,
                          learning_rate=0.01, batch_size=4)

    def grid_search(self, thread=1, start_num=0):
        # TODO
        # activations = ['relu', 'leaky_relu', 'tanh']
        activations = ['relu']
        # unit1s = [32, 64, 128, 256, 516]
        unit1s = [64, 128, 256, 516]
        # unit1s = [32, 64, 128, 256, 516]
        unit2s = [64, 128, 256, 516]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0, 0.2]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.1, 0.01]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [2, 4, 8, 16]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for activation in activations:
            for unit1 in unit1s:
                for unit2 in unit2s:
                    for dropout1 in dropout1s:
                        for dropout2 in dropout2s:
                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    i += 1
                                    if i > start_num:
                                        pool.apply_async(self.build,
                                                         (activation, unit1, unit2, dropout1, dropout2, learning_rate,
                                                          batch_size, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class VARResult(TSResult):
    """
    向量AR(p)： 有趋势，无季节性
    y_t = const_trend + y_t-1 @ w1 + y_t-2 @ w2 + ...
    model consists of {const_trend, w1, w2, ...}, with shape of (1 + lag * width, width)
    grid search:
    """

    def __init__(self, data, lag, width, model_path):
        super().__init__()
        data = np.array(data)
        k = int(0.7 * len(data))
        self.train_data = data[:k]
        self.test_data = data[k:]
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'var/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, p, is_plot=False):
        try:
            model = VAR(self.train_data).fit(maxlags=self.lag, verbose=True, trend='c')
        except ValueError:
            # 数据异常，只能放弃趋势
            model = VAR(self.train_data).fit(maxlags=self.lag, verbose=True, trend='n')
        pre = list(correct_cdf(model.forecast(self.test_data[-self.lag:], steps=1)[0]))
        # 训练集的mse
        # mse = sum([sum([err ** 2 for err in
        #                 (model.forecast(self.data[i:i + self.lag], steps=1) - self.data[i + self.lag]).tolist()[0]])
        #            for i in range(model.fittedvalues.shape[0])]) / model.fittedvalues.size
        # mse = sum(sum(model.resid * model.resid)) / model.fittedvalues.size
        # 验证集的mse
        mse = sum([sum([err ** 2 for err in
                        (model.forecast(self.test_data[i:i + self.lag], steps=1) - self.test_data[i + self.lag])
                       .tolist()[0]])
                   for i in range(self.test_data.shape[0] - self.lag)]) / (
                      (self.test_data.shape[0] - self.lag) * self.width)
        if is_plot:
            plt.plot(self.test_data[-1])
            plt.plot(model.forecast(self.test_data[-self.lag:], steps=1))
            plt.savefig(
                self.model_path + "default_%s.png" % mse)
            plt.close()
        return pre, mse

    def train(self):
        return self.build(None)

    def grid_search(self, thread=1, start_num=0):
        pool = multiprocessing.Pool(processes=thread)
        pool.apply_async(self.build, (None, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class VSARIMAResult(TSResult):
    """
    向量ARIMA
    1. VAR(p)：无趋势或季节性
    2. VARMA(p, q)：有趋势，无季节性
    grid search:
    1. 趋势：pq
    TODO: ERROR: data.shape是[31,100]时报错，但[31,2]时不报错
    grid search:
    """

    def __init__(self, data, lag, width, model_path):
        super().__init__()
        data = np.array(data)
        k = int(0.7 * len(data))
        self.train_data = data[:k]
        self.test_data = data[k:]
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'vsarima/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, p, q, is_plot=False):
        # TODO
        model = VARMAX(self.train_data, order=(p, q),
                       error_cov_type='error_cov_type',
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
        pre = list(correct_cdf(model.forecast(self.test_data[-self.lag:], steps=1)[0]))
        # 训练集的mse
        # mse = sum([sum([err ** 2 for err in
        #                 (model.forecast(self.data[i:i + self.lag], steps=1) - self.data[i + self.lag]).tolist()[0]])
        #            for i in range(model.fittedvalues.shape[0])]) / model.fittedvalues.size
        mse = model.mse
        # 验证集的mse
        # mse = sum([sum([err ** 2 for err in
        #                 (model.forecast(self.test_data[i:i + self.lag], steps=1) - self.test_data[i + self.lag])
        #                .tolist()[0]])
        #            for i in range(self.test_data.shape[0] - self.lag)]) / (
        #               (self.test_data.shape[0] - self.lag) * self.width)
        if is_plot:
            plt.plot(self.test_data[-1])
            plt.plot(model.forecast(self.test_data[-self.lag:], steps=1))
            plt.savefig(
                self.model_path + "%s_%s_%s.png" % (p, q, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(p=2, q=0)

    def grid_search(self, thread=1, start_num=0):
        # TODO
        ps = qs = [0, 1, 2, 3]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for p in ps:
            for q in qs:
                i += 1
                if i > start_num:
                    pool.apply_async(self.build, (p, q, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class FCLSTMResult(TSResult):
    """
    FC-LSTM
    grid search:
    1. 模型结构：激活函数、第一层神经元数量、第一层dropout比例、第二层神经元数量、第二层dropout比例
    2. 训练参数：学习率、批大小
    """

    def __init__(self, data, lag, width, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        data = np.expand_dims(np.array(data), -1)
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i - lag:i] for i in range(lag, len(train_data))])
        self.train_y = np.array([train_data[i] for i in range(lag, len(train_data))])
        self.test_x = np.array([test_data[i - lag:i] for i in range(lag, len(test_data) + 1)])
        self.test_y = np.array([test_data[i] for i in range(lag, len(test_data))])
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'fclstm/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        model = Sequential([
            LSTM(activation=activation, units=unit1, input_shape=(self.lag, self.width), return_sequences=True),
            # Dropout(dropout1),
            LSTM(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=1)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.00001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x[:-1], self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(history.model.predict(self.test_x[-1:])[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum([(pre - true) ** 2 for pre, true in zip(model.predict(self.test_x[:-1]), self.test_y)]))[0] / self.test_y.size
        mse = history.history['val_loss'][-1]
        if is_plot:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0,
                          learning_rate=0.01, batch_size=4)

    def grid_search(self, thread=1, start_num=0):
        # TODO
        # activations = ['relu', 'leaky_relu', 'tanh']
        activations = ['relu']
        # unit1s = [32, 64, 128, 256, 516]
        unit1s = [64, 128, 256, 516]
        # unit1s = [32, 64, 128, 256, 516]
        unit2s = [64, 128, 256, 516]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0, 0.2]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.1, 0.01]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [2, 4, 8, 16]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for activation in activations:
            for unit1 in unit1s:
                for unit2 in unit2s:
                    for dropout1 in dropout1s:
                        for dropout2 in dropout2s:
                            for learning_rate in learning_rates:
                                for batch_size in batch_sizes:
                                    i += 1
                                    if i > start_num:
                                        pool.apply_async(self.build,
                                                         (activation, unit1, unit2, dropout1, dropout2, learning_rate,
                                                          batch_size, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


class ConvLSTMResult(TSResult):
    """
    ConvLSTM
    grid search:
    1. 模型结构：激活函数、第一层神经元数量、第一层dropout比例、第二层神经元数量、第二层dropout比例
    2. 训练参数：学习率、批大小
    """

    def __init__(self, data, lag, width, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        data = np.expand_dims(np.array(data), -1)
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i - lag:i] for i in range(lag, len(train_data))])
        self.train_y = np.array([train_data[i] for i in range(lag, len(train_data))])
        self.test_x = np.array([test_data[i - lag:i] for i in range(lag, len(test_data) + 1)])
        self.test_y = np.array([test_data[i] for i in range(lag, len(test_data))])
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'convlstm/'
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)

    def build(self, activation1, activation2, filter1, filter2, dropout1, dropout2, kernal_size,
              learning_rate, batch_size, is_plot=False):
        """
        filters: 卷积核数量
        kernel_size: 卷积核大小
        strides: 卷积核往右和往下移动的步长
        padding: 处理边界的策略，valid表示不处理边界，输出shape会变小；same表示处理边界，输出shape和输入shape一致
        return_sequences: 是否返回中间序列，true表示输出所有输出，false表示只输出最后一个时间的输出
        """
        start_time = time.time()
        model = Sequential([
            ConvLSTM1D(activation=activation1, filters=filter1, kernel_size=kernal_size, strides=1,
                       input_shape=(self.lag, self.width, 1), padding='same', return_sequences=True),
            # BatchNormalization(),
            Dropout(dropout1),
            ConvLSTM1D(activation=activation1, filters=filter2, kernel_size=kernal_size, strides=1,
                       padding='same', return_sequences=False),
            # BatchNormalization(),
            Dropout(dropout2),
            Conv1D(activation=activation2, filters=1, kernel_size=kernal_size,
                   padding='same', data_format='channels_last')
        ])
        optimizer = Adam(learning_rate=learning_rate)  # TODO:Adam和nadam
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.00001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x[:-1], self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=1)
        pre = list(correct_cdf(model.predict(self.test_x[-1:])[0]))
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum([(pre - true) ** 2 for pre, true in zip(model.predict(self.test_x[:-1]), self.test_y)]))[0] / self.test_y.size
        mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation1, activation2, filter1, filter2, dropout1, dropout2,
                    kernal_size, learning_rate, batch_size, end_time - start_time, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(activation1='tanh', activation2='sigmoid',
                          filter1=16, filter2=32,
                          dropout1=0.0, dropout2=0.0, kernal_size=3,
                          learning_rate=0.01, batch_size=4)

    def grid_search(self, thread=1, start_num=0):
        # activation1s = ['relu', 'leaky_relu', 'tanh']
        activation1s = ['tanh']
        # activation2s = ['relu', 'leaky_relu', 'tanh']
        activation2s = ['tanh']
        # filter1s = [8, 16, 32, 64]
        filter1s = [16]
        # filter2s = [8, 16, 32, 64]
        filter2s = [16]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # kernal_sizes = [3, 6, 9]
        kernal_sizes = [3, 6, 9]
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.01, 0.001, 0.0001]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [4, 1, 16, 64]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for activation1 in activation1s:
            for activation2 in activation2s:
                for filter1 in filter1s:
                    for filter2 in filter2s:
                        for dropout1 in dropout1s:
                            for dropout2 in dropout2s:
                                for kernal_size in kernal_sizes:
                                    for learning_rate in learning_rates:
                                        for batch_size in batch_sizes:
                                            i += 1
                                            if i > start_num:
                                                pool.apply_async(self.build,
                                                                 (activation1, activation2, filter1, filter2,
                                                                  dropout1, dropout2, kernal_size,
                                                                  learning_rate, batch_size, True))
        pool.close()
        pool.join()
        self.print_grid_search_result()


type_model_dict = {
    "es": ESResult,
    "sarima": SARIMAResult,
    "rnn": RNNResult,
    "lstm": LSTMResult,
    "gru": GRUResult,

    "var": VARResult,
    "vsarima": VSARIMAResult,
    "fclstm": FCLSTMResult,
    "convlstm": ConvLSTMResult,
}


def build_cdf(data, cdf_width, key_list):
    x_len = len(data)
    x_max_key = x_len - 1
    cdf = []
    p = 0
    for l in range(cdf_width):
        p = binary_search_less_max_duplicate(data, key_list[l], p, x_max_key)
        cdf.append(p / x_len)
    return cdf


def correct_cdf(cdf):
    """
    correct the predicted cdf:
    1. normalize into [0.0, 1.0]
    2. transfer into monotonically increasing
    e.g. [0.1, 0.0, 0.1, 0.7, 0.6, 1.0, 0.9] => [0.0, 0.0, 0.1, 0.7, 0.7, 1.0, 1.0]
    """
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
    i = 0
    cdf_width = len(cdf)
    cur_v = 0.0
    while i < cdf_width and cdf[i] != 0.0:
        cdf[i] = 0.0
        i += 1
    while i < cdf_width and cdf[i] < 1.0:
        if cdf[i] < cur_v:
            cdf[i] = cur_v
        else:
            cur_v = cdf[i]
        i += 1
    while i < cdf_width:
        cdf[i] = 1.0
        i += 1
    return cdf


def correct_max_key(max_key):
    """
    correct the max_key:
    1. max_key >= 0
    """
    return max(0, round(max_key[0]))
