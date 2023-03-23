import gc
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
from numpy.linalg import LinAlgError
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
        self.cdf_verify_mae = 0
        self.cdf_real_mae = 0
        self.max_key_verify_mae = 0
        self.max_key_real_mae = 0
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
            mae_cdf = 0
            mae_max_key = 0
        elif self.time_id <= lag + predict_step:  # if cdfs are not enough
            pre_cdfs = [self.cdfs[-1] for i in range(predict_step)]
            pre_max_keys = [self.max_keys[-1] for i in range(predict_step)]
            mae_cdf = 0
            mae_max_key = 0
        else:
            ts = sts_model_type[self.model_cdf](self.cdfs[:self.time_id], lag, predict_step, cdf_width, self.model_path)
            # ts.grid_search(thread=4, start_num=0)
            pre_cdfs, mae_cdf = ts.train()
            ts = ts_model_type[self.model_max_key](self.max_keys[:self.time_id], lag, predict_step, self.model_path)
            # ts.grid_search(thread=3, start_num=0)
            pre_max_keys, mae_max_key = ts.train()
            del ts
            gc.collect(generation=0)
        self.cdfs.extend(pre_cdfs)
        self.max_keys.extend(pre_max_keys)
        self.cdf_verify_mae = mae_cdf
        self.max_key_verify_mae = mae_max_key

    def update(self, cur_cdf, cur_max_key, lag, predict_step, cdf_width):
        """
        update with new data and retrain ts model when outdated
        return: retraining nums
        """
        self.cdfs[self.time_id] = cur_cdf
        self.max_keys[self.time_id] = cur_max_key
        self.time_id += 1
        if self.time_id >= len(self.max_keys):
            self.build(lag, predict_step, cdf_width)
            return 1
        return 0


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
        self.data = np.array(data)
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'es/'

    def build(self, trend, seasonal, is_plot=False):
        model = ExponentialSmoothing(self.data, seasonal_periods=self.lag, trend=trend, seasonal=seasonal).fit()
        pre = correct_max_key(model.forecast(steps=self.predict_step)).tolist()
        # mse = model.sse / model.model.nobs
        mae = sum(
            [abs(data) for data in correct_max_key(model.predict(0, self.data.size - 1)) - self.data]) / self.data.size
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(self.data)
            plt.plot(model.predict(0, len(self.data) - 1))
            plt.savefig(
                self.model_path + "%s_%s_%s.png" % (trend, seasonal, mae))
            plt.close()
        return pre, mae

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
        pre = correct_max_key(model.forecast(steps=self.predict_step)).tolist()
        # mse = model.mse
        mae = sum(
            [abs(data) for data in correct_max_key(model.predict(0, self.data.size - 1)) - self.data]) / self.data.size
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(self.data)
            plt.plot(model.predict(0, self.data.size - 1))
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s.png" % (p, d, q, P, D, Q, mae))
            plt.close()
        return pre, mae

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
        group_num = len(data) - lag - predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, k)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, k)])
            self.test_x = np.array([data[i:i + lag] for i in range(k, group_num)])
            self.test_y = np.array([data[i + lag:i + lag + predict_step] for i in range(k, group_num)])
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, group_num)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, group_num)])
            self.test_x = self.train_x
            self.test_y = self.train_y
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
        pre = correct_max_key(model.predict(self.pre_x)[0]).tolist()
        # ERROR: loss里的mse和实际计算的mse有差距
        mae = sum(sum([abs(pre - true)
                       for pre, true in
                       zip(correct_max_key(model.predict(self.test_x)), self.test_y)])) / self.test_y.size
        # mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, mae))
            plt.close()
        return pre, mae

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
        group_num = len(data) - lag - predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, k)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, k)])
            self.test_x = np.array([data[i:i + lag] for i in range(k, group_num)])
            self.test_y = np.array([data[i + lag:i + lag + predict_step] for i in range(k, group_num)])
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, group_num)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, group_num)])
            self.test_x = self.train_x
            self.test_y = self.train_y
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
        pre = correct_max_key(model.predict(self.pre_x)[0]).tolist()
        # ERROR: loss里的mse和实际计算的mse有差距
        mae = sum(sum([abs(pre - true)
                       for pre, true in
                       zip(correct_max_key(model.predict(self.test_x)), self.test_y)])) / self.test_y.size
        # mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, mae))
            plt.close()
        return pre, mae

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
        # learning_rates = [0.01, 0.001, 0.0001]
        learning_rates = [0.1]
        # batch_sizes = [1, 4, 16, 64]
        batch_sizes = [4]
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
        group_num = len(data) - lag - predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, k)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, k)])
            self.test_x = np.array([data[i:i + lag] for i in range(k, group_num)])
            self.test_y = np.array([data[i + lag:i + lag + predict_step] for i in range(k, group_num)])
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, group_num)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, group_num)])
            self.test_x = self.train_x
            self.test_y = self.train_y
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
        pre = correct_max_key(model.predict(self.pre_x)[0]).tolist()
        # ERROR: loss里的mse和实际计算的mse有差距
        mae = sum(sum([abs(pre - true)
                       for pre, true in
                       zip(correct_max_key(model.predict(self.test_x)), self.test_y)])) / self.test_y.size
        # mse = history.history['val_loss'][-1]
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, mae))
            plt.close()
        return pre, mae

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
        if len(data) - k >= lag + predict_step:  # if data is enough, split into train_data and test_data
            self.train_data = data[:k]
            self.test_data = data[k:]
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_data = data
            self.test_data = data
        self.predict_step = predict_step
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'var/'

    def build(self, p, is_plot=False):
        try:
            model = VAR(self.train_data).fit(maxlags=self.lag, verbose=False, trend='c')
        except ValueError:
            # 数据异常，只能放弃趋势
            try:
                model = VAR(self.train_data).fit(maxlags=self.lag, verbose=False, trend='n')
            except LinAlgError:
                model = VAR(self.train_data[1:]).fit(maxlags=self.lag, verbose=False, trend='n')
        pre = correct_cdf(model.forecast(self.test_data[-self.lag:], steps=self.predict_step)).tolist()
        # 训练集的mae
        # train_group_num = len(self.train_data) - self.lag - self.predict_step + 1
        # mae = sum([sum([abs(err)
        #                 for l in (correct_cdf(model.forecast(self.train_data[i:i + self.lag], steps=self.predict_step))
        #                           - self.train_data[self.lag: i + self.lag + self.predict_step])
        #                 for err in l])
        #            for i in range(train_group_num)]) / (train_group_num * self.width * self.predict_step)
        # 验证集的mae
        test_group_num = len(self.test_data) - self.lag - self.predict_step + 1
        mae = sum([sum([abs(err)
                        for l in (correct_cdf(model.forecast(self.test_data[i:i + self.lag], steps=self.predict_step))
                                  - self.test_data[i + self.lag: i + self.lag + self.predict_step])
                        for err in l])
                   for i in range(test_group_num)]) / (test_group_num * self.width * self.predict_step)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(self.test_data[-1])
            plt.plot(model.forecast(self.test_data[-self.lag:], steps=1))
            plt.savefig(
                self.model_path + "default_%s.png" % mae)
            plt.close()
        return pre, mae

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
        if len(data) - k >= lag + predict_step:  # if data is enough, split into train_data and test_data
            self.train_data = data[:k]
            self.test_data = data[k:]
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_data = data
            self.test_data = data
        self.predict_step = predict_step
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'vsarima/'

    def build(self, p, q, is_plot=False):
        model = VARMAX(self.train_data, order=(p, q),
                       error_cov_type='error_cov_type',
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
        pre = correct_cdf(model.forecast(self.test_data[-self.lag:], steps=self.predict_step)).tolist()
        # 训
        # train_group_num = len(self.train_data) - self.lag - self.predict_step + 1
        # mae = sum([sum([abs(err)
        #                 for l in (correct_cdf(model.forecast(self.train_data[i:i + self.lag], steps=self.predict_step))
        #                           - self.train_data[self.lag: i + self.lag + self.predict_step])
        #                 for err in l])
        #            for i in range(train_group_num)]) / (train_group_num * self.width * self.predict_step)
        # mse = model.mse
        # 验证集的mae
        test_group_num = len(self.test_data) - self.lag - self.predict_step + 1
        mae = sum([sum([abs(err)
                        for l in (correct_cdf(model.forecast(self.test_data[i:i + self.lag], steps=self.predict_step))
                                  - self.test_data[i + self.lag: i + self.lag + self.predict_step])
                        for err in l])
                   for i in range(test_group_num)]) / (test_group_num * self.width * self.predict_step)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(self.test_data[-1])
            plt.plot(model.forecast(self.test_data[-self.lag:], steps=1))
            plt.savefig(
                self.model_path + "%s_%s_%s.png" % (p, q, mae))
            plt.close()
        return pre, mae

    def train(self):
        return self.build(p=2, q=0)

    def grid_search(self, thread=1, start_num=0):
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
        group_num = len(data) - lag - predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, k)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, k)])
            self.train_y = self.train_y.reshape(self.train_y.shape[0], predict_step * width)
            self.test_x = np.array([data[i:i + lag] for i in range(k, group_num)])
            self.test_y = np.array([data[i + lag:i + lag + predict_step] for i in range(k, group_num)])
            self.test_y = self.test_y.reshape(self.test_y.shape[0], predict_step * width)
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_x = np.array([data[i:i + lag] for i in range(0, group_num)])
            self.train_y = np.array([data[i + lag:i + lag + predict_step] for i in range(0, group_num)])
            self.train_y = self.train_y.reshape(self.train_y.shape[0], predict_step * width)
            self.test_x = self.train_x
            self.test_y = self.train_y
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
        pre = correct_cdf(model.predict(self.pre_x).reshape(self.predict_step, self.width)).tolist()
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = history.history['val_loss'][-1]
        pres = correct_cdf(model.predict(self.test_x).reshape(self.test_x.shape[0] * self.predict_step, self.width))
        trues = self.test_y.reshape(self.test_x.shape[0] * self.predict_step, self.width)
        mae = np.sum(np.abs(pres - trues)) / self.test_y.size
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, mae))
            plt.close()
        return pre, mae

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
        # learning_rates = [0.1, 0.01, 0.001, 0.0001]
        learning_rates = [0.01]
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

    def __init__(self, data, lag, predict_step, width, model_path):
        super().__init__()
        group_num = len(data) - lag - predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            self.train_x = np.expand_dims(np.array([data[i:i + lag]
                                                    for i in range(0, k)]), -1)
            self.train_y = np.expand_dims(np.array([data[i + lag:i + lag + predict_step]
                                                    for i in range(0, k)]), -1)
            self.test_x = np.expand_dims(np.array([data[i:i + lag]
                                                   for i in range(k, group_num)]), -1)
            self.test_y = np.expand_dims(np.array([data[i + lag:i + lag + predict_step]
                                                   for i in range(k, group_num)]), -1)
        else:  # if data is not enough, keep the same between train_data and test_data
            self.train_x = np.expand_dims(np.array([data[i:i + lag]
                                                    for i in range(0, group_num)]), -1)
            self.train_y = np.expand_dims(np.array([data[i + lag:i + lag + predict_step]
                                                    for i in range(0, group_num)]), -1)
            self.test_x = self.train_x
            self.test_y = self.train_y
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
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.0005, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                            epochs=100, batch_size=batch_size,
                            callbacks=[early_stopping], verbose=0)
        pre = correct_cdf(model.predict(self.pre_x)[0, :, :, 0]).tolist()
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = history.history['val_loss'][-1]
        pres = model.predict(self.test_x).reshape(self.test_x.shape[0] * self.predict_step, self.width)
        trues = self.test_y.reshape(self.test_x.shape[0] * self.predict_step, self.width)
        mae = np.sum(np.abs(pres - trues)) / self.test_y.size
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation1, activation2, filter1, filter2, dropout1, dropout2,
                    kernal_size, learning_rate, batch_size, end_time - start_time, mae))
            plt.close()
        return pre, mae

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
    for i in range(cdfs.shape[0]):
        cdf = cdfs[i]
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        j = 0
        cdf_width = len(cdf)
        cur_v = 0.0
        while j < cdf_width and cdf[j] != 0.0:
            cdf[j] = 0.0
            j += 1
        while j < cdf_width and cdf[j] < 1.0:
            if cdf[j] < cur_v:
                cdf[j] = cur_v
            else:
                cur_v = cdf[j]
            j += 1
        while j < cdf_width:
            cdf[j] = 1.0
            j += 1
        cdfs[i] = cdf
    return cdfs


def correct_max_key(max_keys):
    """
    correct the max_key:
    1. max_key >= 0
    2. transfer into integer
    """
    max_keys = max_keys.astype(np.int)
    max_keys[max_keys < 0] = 0
    return max_keys
