import multiprocessing
import os
import time
import warnings

import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, ConvLSTM1D, SimpleRNN, GRU, Flatten, RepeatVector, Conv2D, Reshape
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


class TimeSeriesModel:
    def __init__(self, key_list, model_path, cdfs, type_cdf, max_keys, type_max_key, data_len):
        # for compute
        self.key_list = key_list
        self.data_len = data_len
        # common
        self.name = "Time Series Model"
        self.model_path = model_path
        self.time_id = len(cdfs)
        # for ts of cdf
        self.cdfs = cdfs
        self.model_cdf = type_cdf
        # for ts of max key
        self.max_keys = max_keys
        self.model_max_key = type_max_key

    def build(self, lag, predict_step, cdf_width):
        if self.time_id == 0:  # if cdfs is []
            pre_cdfs = [[0.0] * cdf_width for i in range(predict_step)]
            pre_max_keys = [0 for i in range(predict_step)]
            mse_cdf = 0
            mse_max_key = 0
        elif self.time_id <= lag + predict_step:  # if cdfs are not enough
            pre_cdfs = [self.cdfs[-1] for i in range(predict_step)]
            pre_max_keys = [self.max_keys[-1] for i in range(predict_step)]
            mse_cdf = 0
            mse_max_key = 0
        else:
            ts = sts_model_type[self.model_cdf](self.cdfs[:self.time_id], lag, predict_step, cdf_width, self.model_path)
            ts.grid_search(thread=1, start_num=0)
            pre_cdfs, mse_cdf = ts.train()
            # ts = ts_model_type[self.model_max_key](self.max_keys[:self.time_id], lag, predict_step, self.model_path)
            # ts.grid_search(thread=3, start_num=0)
            pre_max_keys, mse_max_key = ts.train()
        self.cdfs.extend([])
        self.max_keys.extend([])
        return 0, 0

    def update(self, cur_cdf, cur_max_key, lag, predict_step, cdf_width):
        """
        update with new data and retrain ts model when outdated
        return: retraining nums and mse
        """
        self.cdfs[self.time_id] = cur_cdf
        self.max_keys[self.time_id] = cur_max_key
        self.time_id += 1
        if self.time_id >= len(self.max_keys):
            mse_cdf, mse_max_key = self.build(lag, predict_step, cdf_width)
            return 1, mse_cdf, mse_max_key
        return 0, 0, 0


class TSResult:
    def __init__(self):
        self.model_path = None

    def print_grid_search_result(self):
        files = os.listdir(self.model_path)
        results = [file.rsplit("_", 1) for file in files if file.endswith('.png')]
        for result in results:
            result[1] = float(result[1].replace(".png", ""))
        results.sort(key=lambda x: x[1])
        output_file = open(self.model_path + 'result.txt', 'w')
        template = '%s, %s\n'
        for result in results:
            output_file.write(template % (result[1], result[0]))
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

    def __init__(self, data, lag, predict_step, model_path):
        super().__init__()
        # ES规定数据必须包含不低于两个周期
        if len(data) < 2 * lag:
            data.extend(data[-lag:])
        self.data = data
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'es/'

    def build(self, trend, seasonal, is_plot=False):
        model = ExponentialSmoothing(self.data, seasonal_periods=self.lag, trend=trend, seasonal=seasonal).fit()
        pre = correct_max_key(model.forecast(steps=self.predict_step))
        mse = model.sse / model.model.nobs
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, model_path):
        super().__init__()
        self.data = np.array(data)
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'sarima/'

    def build(self, p, d, q, P, Q, D, is_plot=False):
        model = SARIMAX(self.data, order=(p, d, q), seasonal_order=(P, D, Q, self.lag),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
        pre = correct_max_key(model.forecast(steps=self.predict_step))
        mse = model.mse
        # sum([data ** 2 for data in model.predict(0, self.data.size - 1) - self.data]) / self.data.size
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i:i + lag]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.train_y = np.array([train_data[i + lag:i + lag + predict_step]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.test_x = np.array([test_data[i:i + lag]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.test_y = np.array([test_data[i + lag:i + lag + predict_step]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.pre_x = np.expand_dims(np.array(data[-lag:]), 0)
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'rnn/'

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        model = Sequential([
            SimpleRNN(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            SimpleRNN(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(model.predict(self.pre_x)[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum([(pre - true) ** 2
        #                for pre, true in zip(model.predict(self.test_x), self.test_y)])) / self.test_y.size
        mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i:i + lag]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.train_y = np.array([train_data[i + lag:i + lag + predict_step]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.test_x = np.array([test_data[i:i + lag]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.test_y = np.array([test_data[i + lag:i + lag + predict_step]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.pre_x = np.expand_dims(np.array(data[-lag:]), 0)
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'lstm/'

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        model = Sequential([
            LSTM(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            LSTM(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(model.predict(self.pre_x)[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum([(pre - true) ** 2
        #                for pre, true in zip(model.predict(self.test_x), self.test_y)])) / self.test_y.size
        mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i:i + lag]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.train_y = np.array([train_data[i + lag:i + lag + predict_step]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.test_x = np.array([test_data[i:i + lag]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.test_y = np.array([test_data[i + lag:i + lag + predict_step]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.pre_x = np.expand_dims(np.array(data[-lag:]), 0)
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'gru/'

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        model = Sequential([
            GRU(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            GRU(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_max_key(model.predict(self.pre_x)[0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum([(pre - true) ** 2
        #                for pre, true in zip(model.predict(self.test_x), self.test_y)])) / self.test_y.size
        mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, width, model_path):
        super().__init__()
        data = np.array(data)
        k = int(0.7 * len(data))
        if len(data) - k >= lag + predict_step:  # if data is enough
            self.train_data = data[:k]
            self.test_data = data[k:]
        else:
            self.train_data = data
            self.test_data = data
        self.predict_step = predict_step
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'var/'

    def build(self, p, is_plot=False):
        try:
            model = VAR(self.train_data).fit(maxlags=self.lag, verbose=True, trend='c')
        except ValueError:
            # 数据异常，只能放弃趋势
            model = VAR(self.train_data).fit(maxlags=self.lag, verbose=True, trend='n')
        pre = correct_cdf(model.forecast(self.test_data[-self.lag:], steps=self.predict_step))
        # 训练集的mse
        # train_group_num = len(self.train_data) - self.lag - self.predict_step + 1
        # mse = sum([sum([err ** 2
        #                 for l in (model.forecast(self.train_data[i:i + self.lag], steps=self.predict_step)
        #                           - self.train_data[self.lag: i + self.lag + self.predict_step])
        #                 for err in l])
        #            for i in r*************start tsusli_24_1_100_1_var_es NYCT_SORTED************
        # Build delta model cdf mse: 1.124389254489161e+19
        # Build delta model max_key mse: 500.0360785130816
        # Build time: 577.599601984024
        # Structure size: 269960791
        # Index entry size: 406548800
        # Model precision avg: 1388.291
        # Point query time: 4.188799858093262e-05
        # Point query io cost: 504.989
        # Update time id: 745
        # Insert key time: 0.5170021057128906
        # Insert key io: 20746
        # Delta model cdf mse: 1.9273619781561628
        # Delta model max_key mse: 2026.569
        # Merge data time: 1.250575304031372
        # Merge data io: 99100
        # Retrain delta model num: 399
        # Retrain delta model cdf mse: 1.169868576791633e+19
        # Retrain delta model max_key mse: 502.13132329659703
        # Retrain delta model time: 133.14887642860413
        # Retrain delta model io: 58690.86328125
        # Index entry size: 406900480
        # Model precision avg: 2703.736
        # Update time id: 746
        # Insert key time: 0.31615447998046875
        # Insert key io: 12858
        # Delta model cdf mse: 1.4005387640428615
        # Delta model max_key mse: 70623.257
        # Merge data time: 1.2397773265838623
        # Merge data io: 99392
        # Retrain delta model num: 383
        # Retrain delta model cdf mse: 1.1728772076074818e+19
        # Retrain delta model max_key mse: 502.01953002526517
        # Retrain delta model time: 128.61664247512817
        # Retrain delta model io: 56288.5625
        # Index entry size: 407063076
        # Model precision avg: 2720.931
        # Update time id: 747
        # Insert key time: 0.664823055267334
        # Insert key io: 8302
        # Delta model cdf mse: 1.3222825161313472
        # Delta model max_key mse: 258625.998
        # Merge data time: 1.260871410369873
        # Merge data io: 99616
        # Retrain delta model num: 349
        # Retrain delta model cdf mse: 1.1782096857028528e+19
        # Retrain delta model max_key mse: 503.0893511081118
        # Retrain delta model time: 121.00183820724487
        # Retrain delta model io: 51435.630859375
        # Index entry size: 407210048
        # Model precision avg: 2728.601
        # Update time id: 748
        # Insert key time: 0.15259075164794922
        # Insert key io: 5459
        # Delta model cdf mse: 1.3515175156324986
        # Delta model max_key mse: 379962.73
        # Merge data time: 1.3214654922485352
        # Merge data io: 99668
        # Retrain delta model num: 333
        # Retrain delta model cdf mse: 1.1603290677900542e+19
        # Retrain delta model max_key mse: 502.8164163321437
        # Retrain delta model time: 117.25277996063232
        # Retrain delta model io: 49090.734375
        # Index entry size: 407312948
        # Model precision avg: 2744.091
        # Update time id: 749
        # Insert key time: 0.11070418357849121
        # Insert key io: 4388
        # Delta model cdf mse: 1.8056363970821268
        # Delta model max_key mse: 491576.142
        # Merge data time: 1.251652479171753
        # Merge data io: 99704
        # Retrain delta model num: 334
        # Retrain delta model cdf mse: 1.165596389551232e+19
        # Retrain delta model max_key mse: 502.48857685426043
        # Retrain delta model time: 113.72380948066711
        # Retrain delta model io: 49388.802734375
        # Index entry size: 407446004
        # Model precision avg: 2753.511
        # Update time id: 750
        # Insert key time: 0.12067818641662598
        # Insert key io: 4882
        # Delta model cdf mse: 2.069181630051969
        # Delta model max_key mse: 544635.334
        # Merge data time: 1.2596302032470703
        # Merge data io: 99759
        # Retrain delta model num: 370
        # Retrain delta model cdf mse: 1.2978261831418065e+19
        # Retrain delta model max_key mse: 502.0645106703207
        # Retrain delta model time: 125.59859871864319
        # Retrain delta model io: 54667.0390625
        # Index entry size: 407738184
        # Model precision avg: 2751.318
        # Update time id: 751
        # Insert key time: 0.32015347480773926
        # Insert key io: 12405
        # Delta model cdf mse: 3.2054900191795714
        # Delta model max_key mse: 649465.564
        # Merge data time: 1.284564733505249
        # Merge data io: 99882
        # Retrain delta model num: 393
        # Retrain delta model cdf mse: 1.6109269177555657e+19
        # Retrain delta model max_key mse: 501.7208596884542
        # Retrain delta model time: 133.24758553504944
        # Retrain delta model io: 58213.48046875
        # Index entry size: 408358832
        # Model precision avg: 2690.933
        # Update time id: 752
        # Insert key time: 0.6033844947814941
        # Insert key io: 23343
        # Delta model cdf mse: 4.296921616587366
        # Delta model max_key mse: 812808.186
        # Merge data time: 1.303513526916504
        # Merge data io: 100065
        # Retrain delta model num: 411
        # Retrain delta model cdf mse: 1.5755444897500793e+19
        # Retrain delta model max_key mse: 501.2211340317877
        # Retrain delta model time: 138.72493648529053
        # Retrain delta model io: 60935.943359375
        # Index entry size: 409237780
        # Model precision avg: 2700.189
        # Update time id: 753
        # Insert key time: 0.7679457664489746
        # Insert key io: 28670
        # Delta model cdf mse: 4.279675020805572
        # Delta model max_key mse: 1098747.956
        # Merge data time: 1.3773152828216553
        # Merge data io: 100295
        # Retrain delta model num: 425
        # Retrain delta model cdf mse: 1.5893891068946778e+19
        # Retrain delta model max_key mse: 501.75796366251194
        # Retrain delta model time: 144.88319778442383
        # Retrain delta model io: 62969.16015625
        # Index entry size: 410060672
        # Model precision avg: 2707.799
        # Update time id: 754
        # Insert key time: 0.7310235500335693
        # Insert key io: 27847
        # Delta model cdf mse: 2.289213920682351
        # Delta model max_key mse: 1637435.794
        # Merge data time: 1.3424100875854492
        # Merge data io: 100516
        # Retrain delta model num: 416
        # Retrain delta model cdf mse: 1.623561206617069e+19
        # Retrain delta model max_key mse: 501.26389937698605
        # Retrain delta model time: 140.9320673942566
        # Retrain delta model io: 61681.212890625
        # Index entry size: 410774196
        # Model precision avg: 2715.014
        # Update time id: 755
        # Insert key time: 0.6343233585357666
        # Insert key io: 25107
        # Delta model cdf mse: 2.126932204090485
        # Delta model max_key mse: 2243415.496
        # Merge data time: 1.3733279705047607
        # Merge data io: 100695
        # Retrain delta model num: 403
        # Retrain delta model cdf mse: 1.6693104918867902e+19
        # Retrain delta model max_key mse: 500.74028260930663
        # Retrain delta model time: 135.58842086791992
        # Retrain delta model io: 59934.42578125
        # Index entry size: 411461400
        # Model precision avg: 2726.691
        # Update time id: 756
        # Insert key time: 0.6502416133880615
        # Insert key io: 25266
        # Delta model cdf mse: 2.2005473506044115
        # Delta model max_key mse: 2841635.299
        # Merge data time: 1.341411828994751
        # Merge data io: 100882
        # Retrain delta model num: 416
        # Retrain delta model cdf mse: 1.5171492503719709e+19
        # Retrain delta model max_key mse: 499.7890396814391
        # Retrain delta model time: 141.06268334388733
        # Retrain delta model io: 61860.724609375
        # Index entry size: 412200908
        # Model precision avg: 2729.312
        # Update time id: 757
        # Insert key time: 0.6532597541809082
        # Insert key io: 25707
        # Delta model cdf mse: 2.04630774960727
        # Delta model max_key mse: 3436869.677
        # Merge data time: 1.4820361137390137
        # Merge data io: 101067
        # Retrain delta model num: 403
        # Retrain delta model cdf mse: 1.5307470257541624e+19
        # Retrain delta model max_key mse: 499.7773204829624
        # Retrain delta model time: 138.37387418746948
        # Retrain delta model io: 59938.173828125
        # Index entry size: 412891304
        # Model precision avg: 2737.041
        # Update time id: 758
        # Insert key time: 0.6443185806274414
        # Insert key io: 25358
        # Delta model cdf mse: 1.981285760619929
        # Delta model max_key mse: 4252001.801
        # Merge data time: 1.3463985919952393
        # Merge data io: 101251
        # Retrain delta model num: 413
        # Retrain delta model cdf mse: 1.4723149747615113e+19
        # Retrain delta model max_key mse: 499.44406906572135
        # Retrain delta model time: 141.11353945732117
        # Retrain delta model io: 61509.197265625
        # Index entry size: 413618212
        # Model precision avg: 2740.061
        # Update time id: 759
        # Insert key time: 0.7081050872802734
        # Insert key io: 27096
        # Delta model cdf mse: 2.3329642886358344
        # Delta model max_key mse: 5077133.79
        # Merge data time: 1.3892848491668701
        # Merge data io: 101435
        # Retrain delta model num: 411
        # Retrain delta model cdf mse: 1.4665341476794698e+19
        # Retrain delta model max_key mse: 498.8348507204621
        # Retrain delta model time: 145.38512682914734
        # Retrain delta model io: 61442.32421875
        # Index entry size: 414382472
        # Model precision avg: 2743.023
        # Update time id: 760
        # Insert key time: 0.6482484340667725
        # Insert key io: 24608
        # Delta model cdf mse: 1.950073407550243
        # Delta model max_key mse: 5910721.171
        # Merge data time: 1.396265983581543
        # Merge data io: 101609
        # Retrain delta model num: 407
        # Retrain delta model cdf mse: 1.4617640410208942e+19
        # Retrain delta model max_key mse: 498.5007098886134
        # Retrain delta model time: 139.67509484291077
        # Retrain delta model io: 60827.052734375
        # Index entry size: 414930740
        # Model precision avg: 2745.962
        # Update time id: 761
        # Insert key time: 0.5385606288909912
        # Insert key io: 20923
        # Delta model cdf mse: 1.8839674011709089
        # Delta model max_key mse: 7007905.457
        # Merge data time: 1.4391505718231201
        # Merge data io: 101761
        # Retrain delta model num: 410
        # Retrain delta model cdf mse: 1.4523562735903089e+19
        # Retrain delta model max_key mse: 498.443360816268
        # Retrain delta model time: 144.98027110099792
        # Retrain delta model io: 61335.998046875
        # Index entry size: 415597812
        # Model precision avg: 2748.892
        # Update time id: 762
        # Insert key time: 0.673201322555542
        # Insert key io: 25760
        # Delta model cdf mse: 2.473207456175387
        # Delta model max_key mse: 7966004.887
        # Merge data time: 1.5877594947814941
        # Merge data io: 101938
        # Retrain delta model num: 408
        # Retrain delta model cdf mse: 1.5189013420165425e+19
        # Retrain delta model max_key mse: 498.47490077309146
        # Retrain delta model time: 141.13250064849854
        # Retrain delta model io: 61179.763671875
        # Index entry size: 416511956
        # Model precision avg: 2755.695
        # Update time id: 763
        # Insert key time: 0.8656775951385498
        # Insert key io: 33155
        # Delta model cdf mse: 2.567981438519323
        # Delta model max_key mse: 9122313.385
        # Merge data time: 1.4341633319854736
        # Merge data io: 102168
        # Retrain delta model num: 414
        # Retrain delta model cdf mse: 1.460771761647862e+19
        # Retrain delta model max_key mse: 498.00355086422485
        # Retrain delta model time: 144.4795434474945
        # Retrain delta model io: 62097.048828125
        # Index entry size: 417511164
        # Model precision avg: 2766.41
        # Update time id: 764
        # Insert key time: 0.9364974498748779
        # Insert key io: 35806
        # Delta model cdf mse: 2.577098065893555
        # Delta model max_key mse: 10977162.2
        # Merge data time: 1.5139508247375488
        # Merge data io: 102421
        # Retrain delta model num: 414
        # Retrain delta model cdf mse: 1.5243259691514597e+19
        # Retrain delta model max_key mse: 497.8820561560129
        # Retrain delta model time: 141.87849974632263
        # Retrain delta model io: 62083.634765625
        # Index entry size: 418538624
        # Model precision avg: 2775.538
        # Update time id: 765
        # Insert key time: 1.462090253829956
        # Insert key io: 35254
        # Delta model cdf mse: 2.2070835950586996
        # Delta model max_key mse: 13192296.927
        # Merge data time: 1.5249214172363281
        # Merge data io: 102651
        # Retrain delta model num: 424
        # Retrain delta model cdf mse: 1.3883366331528671e+19
        # Retrain delta model max_key mse: 497.68540619358134
        # Retrain delta model time: 145.21557641029358
        # Retrain delta model io: 63622.8984375
        # Index entry size: 419525008
        # Model precision avg: 2782.243
        # Update time id: 766
        # Insert key time: 1.3693418502807617
        # Insert key io: 34152
        # Delta model cdf mse: 2.2948754754625496
        # Delta model max_key mse: 15900529.086
        # Merge data time: 1.5229289531707764
        # Merge data io: 102888
        # Retrain delta model num: 410
        # Retrain delta model cdf mse: 1.3835251222286481e+19
        # Retrain delta model max_key mse: 498.81035837742195
        # Retrain delta model time: 142.54671359062195
        # Retrain delta model io: 61620.455078125
        # Index entry size: 420427084
        # Model precision avg: 2786.168
        # Update time id: 767
        # Insert key time: 1.3334088325500488
        # Insert key io: 35135
        # Delta model cdf mse: 2.7730584275109105
        # Delta model max_key mse: 19569998.398
        # Merge data time: 1.445134162902832
        # Merge data io: 103141
        # Retrain delta model num: 425
        # Retrain delta model cdf mse: 1.4552062555478512e+19
        # Retrain delta model max_key mse: 498.9520972257096
        # Retrain delta model time: 140.8144211769104
        # Retrain delta model io: 64084.5
        # Index entry size: 421360268
        # Model precision avg: 2804.767
        # Update time id: 768
        # Insert key time: 0.8946177959442139
        # Insert key io: 34487
        # Delta model cdf mse: 2.5340823650979183
        # Delta model max_key mse: 23438115.817
        # Merge data time: 1.4341647624969482
        # Merge data io: 103379
        # Retrain delta model num: 422
        # Retrain delta model cdf mse: 1.4652372420819358e+19
        # Retrain delta model max_key mse: 499.7541364705886
        # Retrain delta model time: 139.16641926765442
        # Retrain delta model io: 63759.998046875
        # Index entry size: 422330804
        # Model precision avg: 2810.738
        # Update time: 3371.13999915123
        # Update io cost: 2982.302
        # Point query time: 4.587697982788086e-05
        # Point query io cost: 2155.273
        # *************start tsusli_24_6_100_1_var_es NYCT_SORTED************
        # Build delta model cdf mse: 2.206656230953027e+112
        # Build delta model max_key mse: 500.0360785130816
        # Build time: 1458.1534433364868
        # Structure size: 274505674
        # Index entry size: 406548800
        # Model precision avg: 1388.291
        # Point query time: 5.2856922149658205e-05
        # Point query io cost: 504.989
        # Update time id: 745
        # Insert key time: 0.5226073265075684
        # Insert key io: 20746
        # Delta model cdf mse: 1.9273619781561628
        # Delta model max_key mse: 2026.569
        # Merge data time: 1.2865591049194336
        # Merge data io: 99100
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.070425033569336
        # Retrain delta model io: 78.708984375
        # Index entry size: 407026228
        # Model precision avg: 2703.736
        # Update time id: 746
        # Insert key time: 0.3211205005645752
        # Insert key io: 12858
        # Delta model cdf mse: 2.623281819523336
        # Delta model max_key mse: 34030.91
        # Merge data time: 1.2526497840881348
        # Merge data io: 99392
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.565739631652832
        # Retrain delta model io: 75.552734375
        # Index entry size: 407312248
        # Model precision avg: 2720.931
        # Update time id: 747
        # Insert key time: 0.215423583984375
        # Insert key io: 8302
        # Delta model cdf mse: 1.884874826617977
        # Delta model max_key mse: 119767.141
        # Merge data time: 1.253648281097412
        # Merge data io: 99616
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.669137001037598
        # Retrain delta model io: 68.845703125
        # Index entry size: 407485204
        # Model precision avg: 2728.601
        # Update time id: 748
        # Insert key time: 0.13563776016235352
        # Insert key io: 5459
        # Delta model cdf mse: 1.7847224607508911
        # Delta model max_key mse: 212321.361
        # Merge data time: 1.2586324214935303
        # Merge data io: 99668
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.395898342132568
        # Retrain delta model io: 65.689453125
        # Index entry size: 407597316
        # Model precision avg: 2744.091
        # Update time id: 749
        # Insert key time: 0.10570979118347168
        # Insert key io: 4388
        # Delta model cdf mse: 2.064621401350187
        # Delta model max_key mse: 299126.23
        # Merge data time: 1.2546441555023193
        # Merge data io: 99704
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.579376220703125
        # Retrain delta model io: 65.88671875
        # Index entry size: 407712508
        # Model precision avg: 2753.511
        # Update time id: 750
        # Insert key time: 0.14162182807922363
        # Insert key io: 4882
        # Delta model cdf mse: 2.905465455950632
        # Delta model max_key mse: 369630.028
        # Merge data time: 1.3204691410064697
        # Merge data io: 99759
        # Retrain delta model num: 262
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 501.9349541581693
        # Retrain delta model time: 202.44848704338074
        # Retrain delta model io: 39093.509765625
        # Index entry size: 407743896
        # Model precision avg: 2751.318
        # Update time id: 751
        # Insert key time: 0.3111567497253418
        # Insert key io: 12405
        # Delta model cdf mse: 3.4659203278244144
        # Delta model max_key mse: 649479.71
        # Merge data time: 1.2755892276763916
        # Merge data io: 99882
        # Retrain delta model num: 45
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 502.08878536123643
        # Retrain delta model time: 42.71873450279236
        # Retrain delta model io: 6779.625
        # Index entry size: 408295104
        # Model precision avg: 2690.933
        # Update time id: 752
        # Insert key time: 0.59836745262146
        # Insert key io: 23343
        # Delta model cdf mse: 18.57510893306088
        # Delta model max_key mse: 877582.182
        # Merge data time: 1.2676074504852295
        # Merge data io: 100065
        # Retrain delta model num: 32
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 502.10284710448224
        # Retrain delta model time: 33.38869023323059
        # Retrain delta model io: 4846.81640625
        # Index entry size: 409084704
        # Model precision avg: 2700.189
        # Update time id: 753
        # Insert key time: 0.734036922454834
        # Insert key io: 28670
        # Delta model cdf mse: 8.956881333983917
        # Delta model max_key mse: 1233554.127
        # Merge data time: 1.276585340499878
        # Merge data io: 100295
        # Retrain delta model num: 22
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 502.10592911537384
        # Retrain delta model time: 27.18034291267395
        # Retrain delta model io: 3360.025390625
        # Index entry size: 409903564
        # Model precision avg: 2707.799
        # Update time id: 754
        # Insert key time: 0.711125373840332
        # Insert key io: 27866
        # Delta model cdf mse: 6.058595473246066
        # Delta model max_key mse: 1717055.181
        # Merge data time: 1.3114917278289795
        # Merge data io: 100516
        # Retrain delta model num: 19
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 502.1132237364444
        # Retrain delta model time: 23.92201280593872
        # Retrain delta model io: 2911.443359375
        # Index entry size: 410647916
        # Model precision avg: 2715.014
        # Update time id: 755
        # Insert key time: 1.0531823635101318
        # Insert key io: 25107
        # Delta model cdf mse: 2.6383718878635487
        # Delta model max_key mse: 2291812.822
        # Merge data time: 1.302515983581543
        # Merge data io: 100695
        # Retrain delta model num: 11
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 502.11361378846243
        # Retrain delta model time: 17.589950799942017
        # Retrain delta model io: 1717.591796875
        # Index entry size: 411366256
        # Model precision avg: 2726.691
        # Update time id: 756
        # Insert key time: 0.6572108268737793
        # Insert key io: 25266
        # Delta model cdf mse: 2.8802047881200488
        # Delta model max_key mse: 2880810.619
        # Merge data time: 1.310495138168335
        # Merge data io: 100882
        # Retrain delta model num: 266
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.7451734677042
        # Retrain delta model time: 206.91952753067017
        # Retrain delta model io: 40002.904296875
        # Index entry size: 412207180
        # Model precision avg: 2729.312
        # Update time id: 757
        # Insert key time: 0.6522712707519531
        # Insert key io: 25707
        # Delta model cdf mse: 2.7552014802697693
        # Delta model max_key mse: 3436716.447
        # Merge data time: 1.3414125442504883
        # Merge data io: 101067
        # Retrain delta model num: 44
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.76319762584984
        # Retrain delta model time: 42.880303144454956
        # Retrain delta model io: 6677.244140625
        # Index entry size: 412938260
        # Model precision avg: 2737.041
        # Update time id: 758
        # Insert key time: 0.635277271270752
        # Insert key io: 25358
        # Delta model cdf mse: 2.0181046652530057
        # Delta model max_key mse: 4133903.588
        # Merge data time: 1.3314390182495117
        # Merge data io: 101251
        # Retrain delta model num: 31
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.7877319193521
        # Retrain delta model time: 33.774659156799316
        # Retrain delta model io: 4723.525390625
        # Index entry size: 413670152
        # Model precision avg: 2740.061
        # Update time id: 759
        # Insert key time: 0.6981513500213623
        # Insert key io: 27096
        # Delta model cdf mse: 2.2987737963779757
        # Delta model max_key mse: 4964098.963
        # Merge data time: 1.3583683967590332
        # Merge data io: 101435
        # Retrain delta model num: 34
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.7510807837217
        # Retrain delta model time: 35.52996325492859
        # Retrain delta model io: 5176.841796875
        # Index entry size: 414414644
        # Model precision avg: 2743.023
        # Update time id: 760
        # Insert key time: 1.059168815612793
        # Insert key io: 24608
        # Delta model cdf mse: 2.036556730957764
        # Delta model max_key mse: 5831240.208
        # Merge data time: 1.3653483390808105
        # Merge data io: 101609
        # Retrain delta model num: 16
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.75587807698076
        # Retrain delta model time: 21.732868671417236
        # Retrain delta model io: 2475.880859375
        # Index entry size: 415021040
        # Model precision avg: 2745.962
        # Update time id: 761
        # Insert key time: 0.5385620594024658
        # Insert key io: 20923
        # Delta model cdf mse: 1.8927454266384136
        # Delta model max_key mse: 6750454.09
        # Merge data time: 1.3703336715698242
        # Merge data io: 101761
        # Retrain delta model num: 11
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.7558413095796
        # Retrain delta model time: 18.69350004196167
        # Retrain delta model io: 1728.44140625
        # Index entry size: 415701832
        # Model precision avg: 2748.892
        # Update time id: 762
        # Insert key time: 0.651254415512085
        # Insert key io: 25733
        # Delta model cdf mse: 2.233143197341919
        # Delta model max_key mse: 7745159.27
        # Merge data time: 1.393273115158081
        # Merge data io: 101938
        # Retrain delta model num: 260
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 498.4764746899105
        # Retrain delta model time: 204.6346390247345
        # Retrain delta model io: 39404.203125
        # Index entry size: 416514280
        # Model precision avg: 2755.695
        # Update time id: 763
        # Insert key time: 0.829805850982666
        # Insert key io: 33155
        # Delta model cdf mse: 3.0301131161113934
        # Delta model max_key mse: 9122110.844
        # Merge data time: 1.3942961692810059
        # Merge data io: 102168
        # Retrain delta model num: 49
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 498.503007195479
        # Retrain delta model time: 47.11397838592529
        # Retrain delta model io: 7486.822265625
        # Index entry size: 417472860
        # Model precision avg: 2766.41
        # Update time id: 764
        # Insert key time: 0.9395449161529541
        # Insert key io: 35806
        # Delta model cdf mse: 3.656979584673877
        # Delta model max_key mse: 11071044.66
        # Merge data time: 1.4231939315795898
        # Merge data io: 102421
        # Retrain delta model num: 30
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 498.4996946566893
        # Retrain delta model time: 32.54395079612732
        # Retrain delta model io: 4608.51953125
        # Index entry size: 418439000
        # Model precision avg: 2775.538
        # Update time id: 765
        # Insert key time: 0.8866267204284668
        # Insert key io: 35254
        # Delta model cdf mse: 5.094694052238705
        # Delta model max_key mse: 13626267.605
        # Merge data time: 1.4381532669067383
        # Merge data io: 102651
        # Retrain delta model num: 28
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 498.5005150345515
        # Retrain delta model time: 30.63705062866211
        # Retrain delta model io: 4297.431640625
        # Index entry size: 419402872
        # Model precision avg: 2782.243
        # Update time id: 766
        # Insert key time: 0.8726644515991211
        # Insert key io: 34273
        # Delta model cdf mse: 2.963619854680829
        # Delta model max_key mse: 16467486.856
        # Merge data time: 1.449125051498413
        # Merge data io: 102888
        # Retrain delta model num: 18
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 498.5028374531695
        # Retrain delta model time: 23.73152232170105
        # Retrain delta model io: 2786.1796875
        # Index entry size: 420338604
        # Model precision avg: 2786.168
        # Update time id: 767
        # Insert key time: 0.8786156177520752
        # Insert key io: 35135
        # Delta model cdf mse: 3.349312084170133
        # Delta model max_key mse: 19774433.592
        # Merge data time: 1.5369033813476562
        # Merge data io: 103141
        # Retrain delta model num: 18
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 498.5048495168558
        # Retrain delta model time: 23.331592082977295
        # Retrain delta model io: 2797.818359375
        # Index entry size: 421216628
        # Model precision avg: 2804.767
        # Update time id: 768
        # Insert key time: 0.8986287117004395
        # Insert key io: 34487
        # Delta model cdf mse: 4.554006699457052
        # Delta model max_key mse: 23919188.911
        # Merge data time: 1.481039047241211
        # Merge data io: 103379
        # Retrain delta model num: 263
        # Retrain delta model cdf mse: 2.206656230953027e+112
        # Retrain delta model max_key mse: 499.68212625803386
        # Retrain delta model time: 206.62635445594788
        # Retrain delta model io: 40159.3359375
        # Index entry size: 422340184
        # Model precision avg: 2810.738
        # Update time: 1415.2284729480743
        # Update io cost: 2982.415
        # Point query time: 4.288530349731445e-05
        # Point query io cost: 2155.266
        # *************start tsusli_24_12_100_1_var_es NYCT_SORTED************
        # Build delta model cdf mse: 1.8622221639579397e+228
        # Build delta model max_key mse: 500.0360785130816
        # Build time: 2332.449461698532
        # Structure size: 279949425
        # Index entry size: 406548800
        # Model precision avg: 1388.291
        # Point query time: 3.9892911911010744e-05
        # Point query io cost: 504.989
        # Update time id: 745
        # Insert key time: 0.527590274810791
        # Insert key io: 20746
        # Delta model cdf mse: 1.9273619781561628
        # Delta model max_key mse: 2026.569
        # Merge data time: 1.2406816482543945
        # Merge data io: 99100
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.382229089736938
        # Retrain delta model io: 78.708984375
        # Index entry size: 407026228
        # Model precision avg: 2703.736
        # Update time id: 746
        # Insert key time: 0.7190954685211182
        # Insert key io: 12858
        # Delta model cdf mse: 2.623281819523336
        # Delta model max_key mse: 34030.91
        # Merge data time: 1.2227342128753662
        # Merge data io: 99392
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.111981391906738
        # Retrain delta model io: 75.552734375
        # Index entry size: 407312248
        # Model precision avg: 2720.931
        # Update time id: 747
        # Insert key time: 0.21143555641174316
        # Insert key io: 8302
        # Delta model cdf mse: 1.884874826617977
        # Delta model max_key mse: 119767.141
        # Merge data time: 1.2486605644226074
        # Merge data io: 99616
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.910786628723145
        # Retrain delta model io: 68.845703125
        # Index entry size: 407485204
        # Model precision avg: 2728.601
        # Update time id: 748
        # Insert key time: 0.15063858032226562
        # Insert key io: 5459
        # Delta model cdf mse: 1.7847224607508911
        # Delta model max_key mse: 212321.361
        # Merge data time: 1.356360912322998
        # Merge data io: 99668
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.384896755218506
        # Retrain delta model io: 65.689453125
        # Index entry size: 407597316
        # Model precision avg: 2744.091
        # Update time id: 749
        # Insert key time: 0.10622620582580566
        # Insert key io: 4388
        # Delta model cdf mse: 2.064621401350187
        # Delta model max_key mse: 299126.23
        # Merge data time: 1.2346975803375244
        # Merge data io: 99704
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.944400310516357
        # Retrain delta model io: 65.88671875
        # Index entry size: 407712508
        # Model precision avg: 2753.511
        # Update time id: 750
        # Insert key time: 0.12267112731933594
        # Insert key io: 4882
        # Delta model cdf mse: 2.905465455950632
        # Delta model max_key mse: 369630.028
        # Merge data time: 1.247664213180542
        # Merge data io: 99759
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.792806148529053
        # Retrain delta model io: 72.98828125
        # Index entry size: 407989092
        # Model precision avg: 2751.318
        # Update time id: 751
        # Insert key time: 0.31316161155700684
        # Insert key io: 12405
        # Delta model cdf mse: 3.3736507360892163
        # Delta model max_key mse: 462475.355
        # Merge data time: 1.2387011051177979
        # Merge data io: 99882
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.680473804473877
        # Retrain delta model io: 77.525390625
        # Index entry size: 408533748
        # Model precision avg: 2690.933
        # Update time id: 752
        # Insert key time: 0.6004424095153809
        # Insert key io: 23343
        # Delta model cdf mse: 18.191531741461215
        # Delta model max_key mse: 665741.277
        # Merge data time: 1.264580488204956
        # Merge data io: 100065
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.810253858566284
        # Retrain delta model io: 81.076171875
        # Index entry size: 409314724
        # Model precision avg: 2700.189
        # Update time id: 753
        # Insert key time: 0.7141180038452148
        # Insert key io: 28670
        # Delta model cdf mse: 8.7050856215486
        # Delta model max_key mse: 989681.312
        # Merge data time: 1.2755787372589111
        # Merge data io: 100295
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.293466567993164
        # Retrain delta model io: 83.837890625
        # Index entry size: 410129552
        # Model precision avg: 2707.799
        # Update time id: 754
        # Insert key time: 1.1130435466766357
        # Insert key io: 27835
        # Delta model cdf mse: 5.656782378601675
        # Delta model max_key mse: 1438752.614
        # Merge data time: 1.358365535736084
        # Merge data io: 100516
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.151845216751099
        # Retrain delta model io: 82.0625
        # Index entry size: 410867464
        # Model precision avg: 2715.014
        # Update time id: 755
        # Insert key time: 0.6263132095336914
        # Insert key io: 25058
        # Delta model cdf mse: 2.2370289626572433
        # Delta model max_key mse: 1982315.347
        # Merge data time: 1.2955350875854492
        # Merge data io: 100695
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.104938507080078
        # Retrain delta model io: 79.498046875
        # Index entry size: 411581856
        # Model precision avg: 2726.691
        # Update time id: 756
        # Insert key time: 0.6412947177886963
        # Insert key io: 25266
        # Delta model cdf mse: 2.1458983360834636
        # Delta model max_key mse: 2547938.547
        # Merge data time: 1.3264539241790771
        # Merge data io: 100882
        # Retrain delta model num: 258
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.5710069273558
        # Retrain delta model time: 315.3066120147705
        # Retrain delta model io: 39117.576171875
        # Index entry size: 412193628
        # Model precision avg: 2729.312
        # Update time id: 757
        # Insert key time: 0.6562747955322266
        # Insert key io: 25707
        # Delta model cdf mse: 2.1264818358798974
        # Delta model max_key mse: 3437008.901
        # Merge data time: 1.3374228477478027
        # Merge data io: 101067
        # Retrain delta model num: 38
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.744433665147
        # Retrain delta model time: 57.03895807266235
        # Retrain delta model io: 5829.001953125
        # Index entry size: 412926444
        # Model precision avg: 2737.041
        # Update time id: 758
        # Insert key time: 0.6562461853027344
        # Insert key io: 25358
        # Delta model cdf mse: 2.0938904979208064
        # Delta model max_key mse: 4134127.894
        # Merge data time: 1.3463993072509766
        # Merge data io: 101251
        # Retrain delta model num: 24
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.78303021943776
        # Retrain delta model time: 40.86369872093201
        # Retrain delta model io: 3712.736328125
        # Index entry size: 413664076
        # Model precision avg: 2740.061
        # Update time id: 759
        # Insert key time: 0.7051277160644531
        # Insert key io: 27096
        # Delta model cdf mse: 2.335119396039168
        # Delta model max_key mse: 4964171.246
        # Merge data time: 1.3643512725830078
        # Merge data io: 101435
        # Retrain delta model num: 29
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.78870853880926
        # Retrain delta model time: 46.268226146698
        # Retrain delta model io: 4468.658203125
        # Index entry size: 414410808
        # Model precision avg: 2743.023
        # Update time id: 760
        # Insert key time: 0.6602051258087158
        # Insert key io: 24608
        # Delta model cdf mse: 2.028865613803449
        # Delta model max_key mse: 5831301.967
        # Merge data time: 1.3862929344177246
        # Merge data io: 101609
        # Retrain delta model num: 14
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.80044937159545
        # Retrain delta model time: 28.671374797821045
        # Retrain delta model io: 2198.525390625
        # Index entry size: 415020816
        # Model precision avg: 2745.962
        # Update time id: 761
        # Insert key time: 0.5405547618865967
        # Insert key io: 20923
        # Delta model cdf mse: 1.9741428189286627
        # Delta model max_key mse: 6750449.252
        # Merge data time: 1.384298324584961
        # Merge data io: 101761
        # Retrain delta model num: 8
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.80099082883487
        # Retrain delta model time: 20.785414457321167
        # Retrain delta model io: 1291.30078125
        # Index entry size: 415701804
        # Model precision avg: 2748.892
        # Update time id: 762
        # Insert key time: 0.6622493267059326
        # Insert key io: 25733
        # Delta model cdf mse: 2.352378382527105
        # Delta model max_key mse: 7745151.261
        # Merge data time: 1.3972628116607666
        # Merge data io: 101938
        # Retrain delta model num: 6
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.79819901580083
        # Retrain delta model time: 18.330968618392944
        # Retrain delta model io: 988.30078125
        # Index entry size: 416567956
        # Model precision avg: 2755.695
        # Update time id: 763
        # Insert key time: 0.8696658611297607
        # Insert key io: 33155
        # Delta model cdf mse: 3.2394356492169303
        # Delta model max_key mse: 9148475.175
        # Merge data time: 1.4571025371551514
        # Merge data io: 102168
        # Retrain delta model num: 5
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.7983333057402
        # Retrain delta model time: 17.517144680023193
        # Retrain delta model io: 838.181640625
        # Index entry size: 417508084
        # Model precision avg: 2766.41
        # Update time id: 764
        # Insert key time: 1.3892788887023926
        # Insert key io: 35806
        # Delta model cdf mse: 3.7471653783111334
        # Delta model max_key mse: 11237505.9
        # Merge data time: 1.4501042366027832
        # Merge data io: 102421
        # Retrain delta model num: 4
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.7973862367657
        # Retrain delta model time: 15.43484902381897
        # Retrain delta model io: 686.484375
        # Index entry size: 418452860
        # Model precision avg: 2775.538
        # Update time id: 765
        # Insert key time: 0.9125766754150391
        # Insert key io: 35254
        # Delta model cdf mse: 5.068553043636441
        # Delta model max_key mse: 13964786.078
        # Merge data time: 1.449125051498413
        # Merge data io: 102651
        # Retrain delta model num: 4
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.7975393809011
        # Retrain delta model time: 15.385844945907593
        # Retrain delta model io: 687.66796875
        # Index entry size: 419397552
        # Model precision avg: 2782.243
        # Update time id: 766
        # Insert key time: 0.9105367660522461
        # Insert key io: 34152
        # Delta model cdf mse: 2.66119218957022
        # Delta model max_key mse: 17008969.798
        # Merge data time: 1.5109586715698242
        # Merge data io: 102888
        # Retrain delta model num: 3
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.79759191804084
        # Retrain delta model time: 14.747541189193726
        # Retrain delta model io: 534.392578125
        # Index entry size: 420316148
        # Model precision avg: 2786.168
        # Update time id: 767
        # Insert key time: 0.9175503253936768
        # Insert key io: 35018
        # Delta model cdf mse: 3.0672309259560624
        # Delta model max_key mse: 20553955.164
        # Merge data time: 1.4949994087219238
        # Merge data io: 103141
        # Retrain delta model num: 2
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.7975856767701
        # Retrain delta model time: 14.220971584320068
        # Retrain delta model io: 386.443359375
        # Index entry size: 421176728
        # Model precision avg: 2804.767
        # Update time id: 768
        # Insert key time: 0.9095757007598877
        # Insert key io: 34498
        # Delta model cdf mse: 3.232646724707996
        # Delta model max_key mse: 24982406.416
        # Merge data time: 1.5717978477478027
        # Merge data io: 103379
        # Retrain delta model num: 253
        # Retrain delta model cdf mse: 1.8622221639579397e+228
        # Retrain delta model max_key mse: 499.69347690101984
        # Retrain delta model time: 331.5247929096222
        # Retrain delta model io: 38952.859375
        # Index entry size: 422338672
        # Model precision avg: 2810.738
        # Update time: 1140.4857921600342
        # Update io cost: 2982.108
        # Point query time: 4.7872304916381835e-05
        # Point query io cost: 2155.27
        # *************start tsusli_24_18_100_1_var_es NYCT_SORTED************
        # Build delta model cdf mse: inf
        # Build delta model max_key mse: 500.0360785130816
        # Build time: 3275.3997831344604
        # Structure size: 285382362
        # Index entry size: 406548800
        # Model precision avg: 1388.291
        # Point query time: 3.989315032958985e-05
        # Point query io cost: 504.989
        # Update time id: 745
        # Insert key time: 0.5066707134246826
        # Insert key io: 20746
        # Delta model cdf mse: 1.9273619781561628
        # Delta model max_key mse: 2026.569
        # Merge data time: 1.297548770904541
        # Merge data io: 99100
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.734288215637207
        # Retrain delta model io: 78.708984375
        # Index entry size: 407026228
        # Model precision avg: 2703.736
        # Update time id: 746
        # Insert key time: 0.32015156745910645
        # Insert key io: 12858
        # Delta model cdf mse: 2.623281819523336
        # Delta model max_key mse: 34030.91
        # Merge data time: 1.2117586135864258
        # Merge data io: 99392
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.814805507659912
        # Retrain delta model io: 75.552734375
        # Index entry size: 407312248
        # Model precision avg: 2720.931
        # Update time id: 747
        # Insert key time: 0.19744658470153809
        # Insert key io: 8302
        # Delta model cdf mse: 1.884874826617977
        # Delta model max_key mse: 119767.141
        # Merge data time: 1.223726749420166
        # Merge data io: 99616
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.950366258621216
        # Retrain delta model io: 68.845703125
        # Index entry size: 407485204
        # Model precision avg: 2728.601
        # Update time id: 748
        # Insert key time: 0.5106263160705566
        # Insert key io: 5459
        # Delta model cdf mse: 1.7847224607508911
        # Delta model max_key mse: 212321.361
        # Merge data time: 1.1967988014221191
        # Merge data io: 99668
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.089720487594604
        # Retrain delta model io: 65.689453125
        # Index entry size: 407597316
        # Model precision avg: 2744.091
        # Update time id: 749
        # Insert key time: 0.10273265838623047
        # Insert key io: 4388
        # Delta model cdf mse: 2.064621401350187
        # Delta model max_key mse: 299126.23
        # Merge data time: 1.191812515258789
        # Merge data io: 99704
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.259251356124878
        # Retrain delta model io: 65.88671875
        # Index entry size: 407712508
        # Model precision avg: 2753.511
        # Update time id: 750
        # Insert key time: 0.11768484115600586
        # Insert key io: 4882
        # Delta model cdf mse: 2.905465455950632
        # Delta model max_key mse: 369630.028
        # Merge data time: 1.2466652393341064
        # Merge data io: 99759
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.682156562805176
        # Retrain delta model io: 72.98828125
        # Index entry size: 407989092
        # Model precision avg: 2751.318
        # Update time id: 751
        # Insert key time: 0.297224760055542
        # Insert key io: 12405
        # Delta model cdf mse: 3.3736507360892163
        # Delta model max_key mse: 462475.355
        # Merge data time: 1.2237451076507568
        # Merge data io: 99882
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.32856822013855
        # Retrain delta model io: 77.525390625
        # Index entry size: 408533748
        # Model precision avg: 2690.933
        # Update time id: 752
        # Insert key time: 0.5773947238922119
        # Insert key io: 23343
        # Delta model cdf mse: 18.191531741461215
        # Delta model max_key mse: 665741.277
        # Merge data time: 1.2426862716674805
        # Merge data io: 100065
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.539807319641113
        # Retrain delta model io: 81.076171875
        # Index entry size: 409314724
        # Model precision avg: 2700.189
        # Update time id: 753
        # Insert key time: 0.7001409530639648
        # Insert key io: 28670
        # Delta model cdf mse: 8.7050856215486
        # Delta model max_key mse: 989681.312
        # Merge data time: 1.252652883529663
        # Merge data io: 100295
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.49621868133545
        # Retrain delta model io: 83.837890625
        # Index entry size: 410129552
        # Model precision avg: 2707.799
        # Update time id: 754
        # Insert key time: 0.6921548843383789
        # Insert key io: 27835
        # Delta model cdf mse: 5.656782378601675
        # Delta model max_key mse: 1438752.614
        # Merge data time: 1.2686069011688232
        # Merge data io: 100516
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.055105686187744
        # Retrain delta model io: 82.0625
        # Index entry size: 410867464
        # Model precision avg: 2715.014
        # Update time id: 755
        # Insert key time: 0.607388973236084
        # Insert key io: 25058
        # Delta model cdf mse: 2.2370289626572433
        # Delta model max_key mse: 1982315.347
        # Merge data time: 1.289551019668579
        # Merge data io: 100695
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.426542520523071
        # Retrain delta model io: 79.498046875
        # Index entry size: 411581856
        # Model precision avg: 2726.691
        # Update time id: 756
        # Insert key time: 0.5953850746154785
        # Insert key io: 25266
        # Delta model cdf mse: 2.1458983360834636
        # Delta model max_key mse: 2547938.547
        # Merge data time: 1.2676095962524414
        # Merge data io: 100882
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.47871446609497
        # Retrain delta model io: 82.0625
        # Index entry size: 412330884
        # Model precision avg: 2729.312
        # Update time id: 757
        # Insert key time: 0.6263585090637207
        # Insert key io: 25707
        # Delta model cdf mse: 2.074015958193052
        # Delta model max_key mse: 3123957.073
        # Merge data time: 1.2676095962524414
        # Merge data io: 101067
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.538869380950928
        # Retrain delta model io: 79.498046875
        # Index entry size: 413060984
        # Model precision avg: 2737.041
        # Update time id: 758
        # Insert key time: 0.6084277629852295
        # Insert key io: 25358
        # Delta model cdf mse: 1.9747858411151111
        # Delta model max_key mse: 3792124.139
        # Merge data time: 1.278580665588379
        # Merge data io: 101251
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.645530700683594
        # Retrain delta model io: 81.470703125
        # Index entry size: 413797160
        # Model precision avg: 2740.061
        # Update time id: 759
        # Insert key time: 0.6671860218048096
        # Insert key io: 27024
        # Delta model cdf mse: 1.8835824898781701
        # Delta model max_key mse: 4591104.305
        # Merge data time: 1.3224635124206543
        # Merge data io: 101435
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.890543937683105
        # Retrain delta model io: 81.076171875
        # Index entry size: 414544172
        # Model precision avg: 2743.023
        # Update time id: 760
        # Insert key time: 0.5834486484527588
        # Insert key io: 24608
        # Delta model cdf mse: 1.7995410566149541
        # Delta model max_key mse: 5428896.506
        # Merge data time: 1.3055083751678467
        # Merge data io: 101609
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.356608152389526
        # Retrain delta model io: 80.287109375
        # Index entry size: 415155244
        # Model precision avg: 2745.962
        # Update time id: 761
        # Insert key time: 0.4976537227630615
        # Insert key io: 20907
        # Delta model cdf mse: 2.2704584448834138
        # Delta model max_key mse: 6319229.337
        # Merge data time: 1.3304412364959717
        # Merge data io: 101761
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.33834719657898
        # Retrain delta model io: 80.87890625
        # Index entry size: 415837408
        # Model precision avg: 2748.892
        # Update time id: 762
        # Insert key time: 0.623345136642456
        # Insert key io: 25733
        # Delta model cdf mse: 2.1937843476518117
        # Delta model max_key mse: 7285324.562
        # Merge data time: 1.3254551887512207
        # Merge data io: 101938
        # Retrain delta model num: 251
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.28914658534273
        # Retrain delta model time: 421.6318905353546
        # Retrain delta model io: 38651.04296875
        # Index entry size: 416509688
        # Model precision avg: 2755.695
        # Update time id: 763
        # Insert key time: 0.8168165683746338
        # Insert key io: 33155
        # Delta model cdf mse: 3.1808416867546274
        # Delta model max_key mse: 9122362.148
        # Merge data time: 1.3912906646728516
        # Merge data io: 102168
        # Retrain delta model num: 43
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.48921102922077
        # Retrain delta model time: 82.12936019897461
        # Retrain delta model io: 6689.474609375
        # Index entry size: 417472524
        # Model precision avg: 2766.41
        # Update time id: 764
        # Insert key time: 0.8806157112121582
        # Insert key io: 35806
        # Delta model cdf mse: 3.0467459134366677
        # Delta model max_key mse: 11071055.946
        # Merge data time: 1.3703351020812988
        # Merge data io: 102421
        # Retrain delta model num: 23
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.52387330146036
        # Retrain delta model time: 47.53784465789795
        # Retrain delta model io: 3616.076171875
        # Index entry size: 418439056
        # Model precision avg: 2775.538
        # Update time id: 765
        # Insert key time: 0.8696913719177246
        # Insert key io: 35254
        # Delta model cdf mse: 3.0390774427453398
        # Delta model max_key mse: 13626250.645
        # Merge data time: 1.3972628116607666
        # Merge data io: 102651
        # Retrain delta model num: 18
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.5298667178581
        # Retrain delta model time: 39.618080377578735
        # Retrain delta model io: 2849.501953125
        # Index entry size: 419402984
        # Model precision avg: 2782.243
        # Update time id: 766
        # Insert key time: 0.8327541351318359
        # Insert key io: 34273
        # Delta model cdf mse: 2.949721787119264
        # Delta model max_key mse: 16467471.324
        # Merge data time: 1.3852946758270264
        # Merge data io: 102888
        # Retrain delta model num: 12
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.5437535588518
        # Retrain delta model time: 29.16416049003601
        # Retrain delta model io: 1924.91796875
        # Index entry size: 420338800
        # Model precision avg: 2786.168
        # Update time id: 767
        # Insert key time: 0.9165909290313721
        # Insert key io: 35135
        # Delta model cdf mse: 3.3937013437981225
        # Delta model max_key mse: 19774418.705
        # Merge data time: 1.4431402683258057
        # Merge data io: 103141
        # Retrain delta model num: 13
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.5468098058442
        # Retrain delta model time: 33.339895725250244
        # Retrain delta model io: 2081.546875
        # Index entry size: 421217132
        # Model precision avg: 2804.767
        # Update time id: 768
        # Insert key time: 0.8477189540863037
        # Insert key io: 34487
        # Delta model cdf mse: 4.667970768809357
        # Delta model max_key mse: 23919169.945
        # Merge data time: 1.4142179489135742
        # Merge data io: 103379
        # Retrain delta model num: 7
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 498.54661069423446
        # Retrain delta model time: 23.346805572509766
        # Retrain delta model io: 1158.935546875
        # Index entry size: 422076200
        # Model precision avg: 2810.738
        # Update time: 936.2535433769226
        # Update io cost: 2982.247
        # Point query time: 4.587721824645996e-05
        # Point query io cost: 2155.284
        # *************start tsusli_24_24_100_1_var_es NYCT_SORTED************
        # Build delta model cdf mse: inf
        # Build delta model max_key mse: 500.0360785130816
        # Build time: 4083.138402700424
        # Structure size: 290782741
        # Index entry size: 406548800
        # Model precision avg: 1388.291
        # Point query time: 4.288458824157715e-05
        # Point query io cost: 504.989
        # Update time id: 745
        # Insert key time: 0.5255687236785889
        # Insert key io: 20746
        # Delta model cdf mse: 1.9273619781561628
        # Delta model max_key mse: 2026.569
        # Merge data time: 1.2775828838348389
        # Merge data io: 99100
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.948787689208984
        # Retrain delta model io: 78.708984375
        # Index entry size: 407026228
        # Model precision avg: 2703.736
        # Update time id: 746
        # Insert key time: 0.7280807495117188
        # Insert key io: 12858
        # Delta model cdf mse: 2.623281819523336
        # Delta model max_key mse: 34030.91
        # Merge data time: 1.2536461353302002
        # Merge data io: 99392
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.467003107070923
        # Retrain delta model io: 75.552734375
        # Index entry size: 407312248
        # Model precision avg: 2720.931
        # Update time id: 747
        # Insert key time: 0.2174375057220459
        # Insert key io: 8302
        # Delta model cdf mse: 1.884874826617977
        # Delta model max_key mse: 119767.141
        # Merge data time: 1.3194708824157715
        # Merge data io: 99616
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.47996997833252
        # Retrain delta model io: 68.845703125
        # Index entry size: 407485204
        # Model precision avg: 2728.601
        # Update time id: 748
        # Insert key time: 0.14660859107971191
        # Insert key io: 5459
        # Delta model cdf mse: 1.7847224607508911
        # Delta model max_key mse: 212321.361
        # Merge data time: 1.30950927734375
        # Merge data io: 99668
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.092005491256714
        # Retrain delta model io: 65.689453125
        # Index entry size: 407597316
        # Model precision avg: 2744.091
        # Update time id: 749
        # Insert key time: 0.11170125007629395
        # Insert key io: 4388
        # Delta model cdf mse: 2.064621401350187
        # Delta model max_key mse: 299126.23
        # Merge data time: 1.372328758239746
        # Merge data io: 99704
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 9.835691213607788
        # Retrain delta model io: 65.88671875
        # Index entry size: 407712508
        # Model precision avg: 2753.511
        # Update time id: 750
        # Insert key time: 0.54852294921875
        # Insert key io: 4882
        # Delta model cdf mse: 2.905465455950632
        # Delta model max_key mse: 369630.028
        # Merge data time: 1.3204681873321533
        # Merge data io: 99759
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.287989139556885
        # Retrain delta model io: 72.98828125
        # Index entry size: 407989092
        # Model precision avg: 2751.318
        # Update time id: 751
        # Insert key time: 0.33011531829833984
        # Insert key io: 12405
        # Delta model cdf mse: 3.3736507360892163
        # Delta model max_key mse: 462475.355
        # Merge data time: 1.3005216121673584
        # Merge data io: 99882
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.16713571548462
        # Retrain delta model io: 77.525390625
        # Index entry size: 408533748
        # Model precision avg: 2690.933
        # Update time id: 752
        # Insert key time: 0.6163549423217773
        # Insert key io: 23343
        # Delta model cdf mse: 18.191531741461215
        # Delta model max_key mse: 665741.277
        # Merge data time: 1.3723399639129639
        # Merge data io: 100065
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.364601373672485
        # Retrain delta model io: 81.076171875
        # Index entry size: 409314724
        # Model precision avg: 2700.189
        # Update time id: 753
        # Insert key time: 0.7360892295837402
        # Insert key io: 28670
        # Delta model cdf mse: 8.7050856215486
        # Delta model max_key mse: 989681.312
        # Merge data time: 1.3483936786651611
        # Merge data io: 100295
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.191065073013306
        # Retrain delta model io: 83.837890625
        # Index entry size: 410129552
        # Model precision avg: 2707.799
        # Update time id: 754
        # Insert key time: 0.7300827503204346
        # Insert key io: 27835
        # Delta model cdf mse: 5.656782378601675
        # Delta model max_key mse: 1438752.614
        # Merge data time: 1.3334338665008545
        # Merge data io: 100516
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.567732095718384
        # Retrain delta model io: 82.0625
        # Index entry size: 410867464
        # Model precision avg: 2715.014
        # Update time id: 755
        # Insert key time: 0.689180850982666
        # Insert key io: 25058
        # Delta model cdf mse: 2.2370289626572433
        # Delta model max_key mse: 1982315.347
        # Merge data time: 1.4531140327453613
        # Merge data io: 100695
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.188072204589844
        # Retrain delta model io: 79.498046875
        # Index entry size: 411581856
        # Model precision avg: 2726.691
        # Update time id: 756
        # Insert key time: 0.6582529544830322
        # Insert key io: 25266
        # Delta model cdf mse: 2.1458983360834636
        # Delta model max_key mse: 2547938.547
        # Merge data time: 1.4192028045654297
        # Merge data io: 100882
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.567095756530762
        # Retrain delta model io: 82.0625
        # Index entry size: 412330884
        # Model precision avg: 2729.312
        # Update time id: 757
        # Insert key time: 0.6732087135314941
        # Insert key io: 25707
        # Delta model cdf mse: 2.074015958193052
        # Delta model max_key mse: 3123957.073
        # Merge data time: 1.393273115158081
        # Merge data io: 101067
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.236943006515503
        # Retrain delta model io: 79.498046875
        # Index entry size: 413060984
        # Model precision avg: 2737.041
        # Update time id: 758
        # Insert key time: 0.6522433757781982
        # Insert key io: 25358
        # Delta model cdf mse: 1.9747858411151111
        # Delta model max_key mse: 3792124.139
        # Merge data time: 1.3723297119140625
        # Merge data io: 101251
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.648520946502686
        # Retrain delta model io: 81.470703125
        # Index entry size: 413797160
        # Model precision avg: 2740.061
        # Update time id: 759
        # Insert key time: 0.685164213180542
        # Insert key io: 27024
        # Delta model cdf mse: 1.8835824898781701
        # Delta model max_key mse: 4591104.305
        # Merge data time: 1.3892829418182373
        # Merge data io: 101435
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.09731674194336
        # Retrain delta model io: 81.076171875
        # Index entry size: 414544172
        # Model precision avg: 2743.023
        # Update time id: 760
        # Insert key time: 0.623333215713501
        # Insert key io: 24608
        # Delta model cdf mse: 1.7995410566149541
        # Delta model max_key mse: 5428896.506
        # Merge data time: 1.4501211643218994
        # Merge data io: 101609
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.042479038238525
        # Retrain delta model io: 80.287109375
        # Index entry size: 415155244
        # Model precision avg: 2745.962
        # Update time id: 761
        # Insert key time: 0.5405492782592773
        # Insert key io: 20907
        # Delta model cdf mse: 2.2704584448834138
        # Delta model max_key mse: 6319229.337
        # Merge data time: 1.4112250804901123
        # Merge data io: 101761
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.300808906555176
        # Retrain delta model io: 80.87890625
        # Index entry size: 415837408
        # Model precision avg: 2748.892
        # Update time id: 762
        # Insert key time: 0.6861636638641357
        # Insert key io: 25733
        # Delta model cdf mse: 2.1937843476518117
        # Delta model max_key mse: 7285324.562
        # Merge data time: 1.4192051887512207
        # Merge data io: 101938
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.187082529067993
        # Retrain delta model io: 80.484375
        # Index entry size: 416703252
        # Model precision avg: 2755.695
        # Update time id: 763
        # Insert key time: 0.8567345142364502
        # Insert key io: 33155
        # Delta model cdf mse: 2.983763180787762
        # Delta model max_key mse: 8651002.557
        # Merge data time: 1.4571020603179932
        # Merge data io: 102168
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.22896409034729
        # Retrain delta model io: 81.66796875
        # Index entry size: 417643996
        # Model precision avg: 2766.41
        # Update time id: 764
        # Insert key time: 0.9165213108062744
        # Insert key io: 35806
        # Delta model cdf mse: 2.729555231178008
        # Delta model max_key mse: 10688423.47
        # Merge data time: 1.4471251964569092
        # Merge data io: 102421
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 10.812078714370728
        # Retrain delta model io: 81.66796875
        # Index entry size: 418589696
        # Model precision avg: 2775.538
        # Update time id: 765
        # Insert key time: 0.8895363807678223
        # Insert key io: 35254
        # Delta model cdf mse: 2.79578294960989
        # Delta model max_key mse: 13354698.573
        # Merge data time: 1.4590981006622314
        # Merge data io: 102651
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.472313642501831
        # Retrain delta model io: 83.640625
        # Index entry size: 419535116
        # Model precision avg: 2782.243
        # Update time id: 766
        # Insert key time: 0.9165594577789307
        # Insert key io: 34751
        # Delta model cdf mse: 2.498395239580092
        # Delta model max_key mse: 16338236.017
        # Merge data time: 1.5787756443023682
        # Merge data io: 102888
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.53514575958252
        # Retrain delta model io: 80.87890625
        # Index entry size: 420453432
        # Model precision avg: 2786.168
        # Update time id: 767
        # Insert key time: 0.9554665088653564
        # Insert key io: 34882
        # Delta model cdf mse: 2.6022471638858473
        # Delta model max_key mse: 19819714.475
        # Merge data time: 1.7423408031463623
        # Merge data io: 103141
        # Retrain delta model num: 0
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 500.0360785130816
        # Retrain delta model time: 11.881219863891602
        # Retrain delta model io: 83.837890625
        # Index entry size: 421313704
        # Model precision avg: 2804.767
        # Update time id: 768
        # Insert key time: 0.9773554801940918
        # Insert key io: 34441
        # Delta model cdf mse: 2.991814879162285
        # Delta model max_key mse: 24175890.878
        # Merge data time: 1.743337631225586
        # Merge data io: 103379
        # Retrain delta model num: 250
        # Retrain delta model cdf mse: inf
        # Retrain delta model max_key mse: 499.4664027511445
        # Retrain delta model time: 556.1898763179779
        # Retrain delta model io: 39091.931640625
        # Index entry size: 422333520
        # Model precision avg: 2810.738
        # Update time: 902.1691839694977
        # Update io cost: 2982.426
        # Point query time: 4.886889457702637e-05
        # Point query io cost: 2155.263
        # *************start tsusli_24_1_100_1_var_es NYCT_SORTED************
        # Build delta model cdf mse: 1.124389254489161e+19
        # Build delta model max_key mse: 500.0360785130816
        # Build time: 744.9026973247528
        # Structure size: 269960791
        # Index entry size: 406548800
        # Model precision avg: 1388.291
        # Point query time: 7.579612731933593e-05
        # Point query io cost: 504.989
        # Update time id: 745
        # Insert key time: 1.0083246231079102
        # Insert key io: 20746
        # Delta model cdf mse: 1.9273619781561628
        # Delta model max_key mse: 2026.569
        # Merge data time: 7.899869918823242
        # Merge data io: 99100
        # *************start tsusli_24_1_100_1_var_es NYCT_SORTED************
        # *************start tsusli_24_1_100_1_var_es NYCT_SORTED************ange(train_group_num)]) / (train_group_num * self.width * self.predict_step)
        # 验证集的mse
        test_group_num = len(self.test_data) - self.lag - self.predict_step + 1
        mse = sum([sum([err
                        for l in (model.forecast(self.test_data[i:i + self.lag], steps=self.predict_step)
                                  - self.test_data[i + self.lag: i + self.lag + self.predict_step])
                        for err in l])
                   for i in range(test_group_num)]) / (test_group_num * self.width * self.predict_step)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, width, model_path):
        super().__init__()
        data = np.array(data)
        k = int(0.7 * len(data))
        if len(data) - k >= lag + predict_step:  # if data is enough
            self.train_data = data[:k]
            self.test_data = data[k:]
        else:
            self.train_data = data
            self.test_data = data
        self.predict_step = predict_step
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'vsarima/'

    def build(self, p, q, is_plot=False):
        # TODO
        model = VARMAX(self.train_data, order=(p, q),
                       error_cov_type='error_cov_type',
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
        pre = list(correct_cdf(model.forecast(self.test_data[-self.lag:], steps=1)[0]))
        # 训练集的mse
        # train_group_num = len(self.train_data) - self.lag - self.predict_step + 1
        # mse = sum([sum([err ** 2
        #                 for l in (model.forecast(self.train_data[i:i + self.lag], steps=self.predict_step)
        #                           - self.train_data[self.lag: i + self.lag + self.predict_step])
        #                 for err in l])
        #            for i in range(train_group_num)]) / (train_group_num * self.width * self.predict_step)
        # mse = model.mse
        # 验证集的mse
        test_group_num = len(self.test_data) - self.lag - self.predict_step + 1
        mse = sum([sum([err ** 2
                        for l in (model.forecast(self.test_data[i:i + self.lag], steps=self.predict_step)
                                  - self.test_data[i + self.lag: i + self.lag + self.predict_step])
                        for err in l])
                   for i in range(test_group_num)]) / (test_group_num * self.width * self.predict_step)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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

    def __init__(self, data, lag, predict_step, width, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i:i + lag]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.train_y = np.array([train_data[i + lag:i + lag + predict_step]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.train_y = self.train_y.reshape(self.train_y.shape[0], predict_step * width)
        self.test_x = np.array([test_data[i:i + lag]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.test_y = np.array([test_data[i + lag:i + lag + predict_step]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.test_y = self.test_y.reshape(self.test_y.shape[0], predict_step * width)
        self.pre_x = np.expand_dims(np.array(data[-lag:]), 0)
        self.lag = lag
        self.predict_step = predict_step
        self.width = width
        self.model_path = model_path + 'fclstm/'

    def build(self, activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        model = Sequential([
            LSTM(activation=activation, units=unit1, input_shape=(self.lag, self.width), return_sequences=True),
            # Dropout(dropout1),
            LSTM(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step * self.width)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.0005, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_cdf(model.predict(self.pre_x).reshape(self.predict_step, self.width))
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum(sum([(pre - true) ** 2 for pre, true in
        #                    zip(model.predict(self.test_x), self.test_y[:, :, :, 0])]))) / self.test_y.size
        mse = history.history['val_loss'][-1]
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
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
        # activations = ['relu', 'leaky_relu', 'tanh']
        activations = ['relu']
        # unit1s = [32, 64, 128, 256, 516]
        unit1s = [64, 128, 256, 516]
        # unit1s = [32, 64, 128, 256, 516]
        unit2s = [64, 128, 256, 516]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # learning_rates = [0.1, 0.01, 0.001, 0.0001]
        learning_rates = [0.01]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [4]
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
                                        self.build(activation, unit1, unit2, dropout1, dropout2, learning_rate,
                                                   batch_size, True)
        self.print_grid_search_result()


class ConvLSTMResult(TSResult):
    """
    ConvLSTM
    grid search:
    1. 模型结构：激活函数、第一层神经元数量、第一层dropout比例、第二层神经元数量、第二层dropout比例
    2. 训练参数：学习率、批大小
    """

    def __init__(self, data, lag, predict_step, width, model_path):
        super().__init__()
        k = int(0.7 * len(data))
        data = np.expand_dims(np.array(data), -1)
        train_data, test_data = data[:k], data[k:]
        self.train_x = np.array([train_data[i:i + lag]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.train_y = np.array([train_data[i + lag:i + lag + predict_step]
                                 for i in range(0, k - lag - predict_step + 1)])
        self.test_x = np.array([test_data[i:i + lag]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.test_y = np.array([test_data[i + lag:i + lag + predict_step]
                                for i in range(0, len(test_data) - lag - predict_step + 1)])
        self.pre_x = np.expand_dims(np.array(data[-lag:]), 0)
        self.lag = lag
        self.predict_step = predict_step
        self.width = width
        self.model_path = model_path + 'convlstm/'

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
        # 1. ConvLSTM编码-LSTM+Dense解码
        # ConvLSTM1D编码，Flatten压扁后RepeatVector重复predict_step次，LSTM给重复次数之间施加时间特征，Dense还原每次的shape
        # model = Sequential([
        #     ConvLSTM1D(activation=activation1, filters=filter1, kernel_size=kernal_size, strides=1,
        #                input_shape=(self.lag, self.width, 1), padding='same', return_sequences=True),
        #     ConvLSTM1D(activation=activation1, filters=filter1, kernel_size=kernal_size, strides=1,
        #                padding='same', return_sequences=False),
        #     # BatchNormalization(),
        #     Dropout(dropout1),
        #     Flatten(),
        #     RepeatVector(self.predict_step),
        #     LSTM(128, activation=activation2, return_sequences=True),
        #     TimeDistributed(Dense(self.width))
        # ])

        # 3. ConvLSTM编码-Reshape+Conv2D解码
        model = Sequential([
            ConvLSTM1D(activation=activation1, filters=filter1, kernel_size=kernal_size, strides=1,
                       input_shape=(self.lag, self.width, 1), padding='same', return_sequences=False),
            # BatchNormalization(),
            # Dropout(dropout1),
            Flatten(),
            RepeatVector(self.predict_step),
            Reshape((self.predict_step, self.width, filter1)),
            ConvLSTM1D(activation=activation1, filters=filter2, kernel_size=kernal_size, strides=1,
                       padding='same', return_sequences=True),
            # BatchNormalization(),
            # Dropout(dropout2),
            Conv2D(activation=activation2, filters=1, kernel_size=(kernal_size, kernal_size),
                   padding='same', data_format='channels_last')
        ])
        optimizer = Adam(learning_rate=learning_rate)  # TODO:Adam和nadam
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.0005, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=1)
        pre = correct_cdf(model.predict(self.pre_x)[0, :, :, 0])
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = sum(sum(sum([(pre - true) ** 2 for pre, true in
        #                    zip(model.predict(self.test_x), self.test_y[:, :, :, 0])]))) / self.test_y.size
        mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation1, activation2, filter1, filter2, dropout1, dropout2,
                    kernal_size, learning_rate, batch_size, end_time - start_time, mse))
            plt.close()
        return pre, mse

    def train(self):
        return self.build(activation1='tanh', activation2='tanh',
                          filter1=8, filter2=8,
                          dropout1=0.0, dropout2=0.0, kernal_size=9,
                          learning_rate=0.01, batch_size=32)

    def grid_search(self, thread=1, start_num=0):
        # activation1s = ['relu', 'leaky_relu', 'tanh']
        activation1s = ['tanh']
        # activation2s = ['relu', 'leaky_relu', 'tanh']
        activation2s = ['tanh']
        # filter1s = [8, 16, 32, 64]
        filter1s = [8]
        # filter2s = [8, 16, 32, 64]
        filter2s = [8]
        # dropout1s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout1s = [0.0]
        # dropout2s = [0.0, 0.2, 0.4, 0.6, 0.8]
        dropout2s = [0.0]
        # kernal_sizes = [3, 6, 9, 12, 15]
        kernal_sizes = [9]
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.01]  # lr0.001对应bs2-4，lr0.01对应bs16-32
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [32]
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


ts_model_type = {
    "es": ESResult,
    "sarima": SARIMAResult,
    "rnn": RNNResult,
    "lstm": LSTMResult,
    "gru": GRUResult,
}

sts_model_type = {
    "var": VARResult,
    "vsarima": VSARIMAResult,
    "fclstm": FCLSTMResult,
    "convlstm": ConvLSTMResult,
}


def correct_cdf(cdfs):
    """
    correct the predicted cdf:
    1. normalize into [0.0, 1.0]
    2. transfer into monotonically increasing
    e.g. [0.1, 0.0, 0.1, 0.7, 0.6, 1.0, 0.9] => [0.0, 0.0, 0.1, 0.7, 0.7, 1.0, 1.0]
    """
    result_cdfs = [None] * cdfs.shape[0]
    j = 0
    for cdf in cdfs:
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
        result_cdfs[j] = cdf.tolist()
        j += 1
    return result_cdfs


def correct_max_key(max_keys):
    """
    correct the max_key:
    1. max_key >= 0
    """
    return [max(0, round(max_key)) for max_key in max_keys]
