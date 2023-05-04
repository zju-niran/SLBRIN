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
    """
    时空序列预测模型：statsmodels或Keras实现，服务于TSUM
    1. F（cdf）的预测模型是时空序列预测（Spatial-Temporal Series Model，sts_model）
    2. n（max_key）的预测模型是时空序列预测（Temporal Series Model，ts_model）
    两者分别由多种不同的预测算法实现（TSResult的实现类）
    """

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
        self.sts_model = None
        # for ts of max key
        self.max_keys = max_keys
        self.model_max_key = type_max_key
        self.ts_model = None

    def build(self, lag, predict_step, cdf_width, threshold_err_cdf=0, threshold_err_max_key=0):
        num_cdf = 0
        num_max_key = 0
        if self.time_id == 0:  # if cdfs is []
            pre_cdfs = [[0.0] * cdf_width for i in range(predict_step)]
            pre_max_keys = [0 for i in range(predict_step)]
        elif self.time_id <= lag + predict_step:  # if cdfs are not enough
            pre_cdfs = [self.cdfs[-1] for i in range(predict_step)]
            pre_max_keys = [self.max_keys[-1] for i in range(predict_step)]
        else:
            if self.sts_model:
                pre_cdfs, num_cdf = self.sts_model.retrain(self.cdfs[:self.time_id], threshold_err_cdf)
            else:
                self.sts_model = sts_model_type[self.model_cdf](lag, predict_step, cdf_width, self.model_path)
                # self.sts_model.grid_search(thread=4, start_num=0)
                pre_cdfs, num_cdf = self.sts_model.train(self.cdfs[:self.time_id])
            if self.ts_model:
                pre_max_keys, num_max_key = self.ts_model.retrain(self.max_keys[:self.time_id], threshold_err_max_key)
            else:
                self.ts_model = ts_model_type[self.model_max_key](lag, predict_step, self.model_path)
                # self.ts_model.grid_search(thread=3, start_num=0)
                pre_max_keys, num_max_key = self.ts_model.train(self.max_keys[:self.time_id])
        self.cdfs.extend(pre_cdfs)
        self.max_keys.extend(pre_max_keys)
        self.cdf_verify_mae = self.sts_model.err if self.sts_model else 0
        self.max_key_verify_mae = self.ts_model.err if self.ts_model else 0
        return num_cdf, num_max_key

    def update(self, cur_cdf, cur_max_key, lag, predict_step, cdf_width, threshold_err_cdf, threshold_err_max_key):
        """
        update with new data and retrain ts model when outdated
        return: retraining nums
        """
        self.cdfs[self.time_id] = cur_cdf
        self.max_keys[self.time_id] = cur_max_key
        self.time_id += 1
        if self.time_id >= len(self.max_keys):
            return self.build(lag, predict_step, cdf_width, threshold_err_cdf, threshold_err_max_key)
        return 0, 0


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

    def __init__(self, lag, predict_step, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'es/'
        self.model = None
        self.err = 0

    def init_data(self, data):
        # ES规定数据必须包含不低于两个周期
        if len(data) < 2 * self.lag:
            data.extend(data[-self.lag:])
        return np.array(data)

    def build(self, data, trend, seasonal, is_plot=False):
        self.model = ExponentialSmoothing(data, seasonal_periods=self.lag, trend=trend, seasonal=seasonal).fit()
        self.err = self.get_err(data)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(data)
            plt.plot(self.model.predict(0, data.size - 1))
            plt.savefig(
                self.model_path + "%s_%s_%s.png" % (trend, seasonal, self.err))
            plt.close()

    def predict(self):
        return correct_max_key(self.model.forecast(steps=self.predict_step)).tolist()

    def train(self, data):
        data = self.init_data(data)
        self.build(data, trend='add', seasonal='add')
        return self.predict(), 1

    def retrain(self, data, threshold_err):
        data = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(data)
        if self.err <= threshold_err * old_err:
            return self.predict(), 0
        else:
            self.build(data, trend='add', seasonal='add')
            return self.predict(), 1

    def get_err(self, data):
        # mse = model.sse / model.model.nobs
        mae = sum([abs(err)
                   for err in correct_max_key(self.model.predict(0, data.size - 1)) - data]) / data.size
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        data = self.init_data(data)
        trends = ["add", "mul", "additive", "multiplicative", None]
        seasonals = ["add", "mul", "additive", "multiplicative", None]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for trend in trends:
            for seasonal in seasonals:
                i += 1
                if i > start_num:
                    pool.apply_async(self.build, (data, trend, seasonal, True))
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

    def __init__(self, lag, predict_step, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'sarima/'
        self.model = None
        self.err = 0

    def init_data(self, data):
        return np.array(data)

    def build(self, data, p, d, q, P, Q, D, is_plot=False):
        self.model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, self.lag),
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)
        self.err = self.get_err(data)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(data)
            plt.plot(self.model.predict(0, data.size - 1))
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s.png" % (p, d, q, P, D, Q, self.err))
            plt.close()

    def predict(self):
        return correct_max_key(self.model.forecast(steps=self.predict_step)).tolist()

    def train(self, data):
        data = self.init_data(data)
        self.build(data, p=3, d=1, q=0, P=2, D=0, Q=3)
        return self.predict(), 1

    def retrain(self, data, threshold_err):
        data = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(data)
        if self.err <= threshold_err * old_err:
            return self.predict(), 0
        else:
            self.build(data, p=3, d=1, q=0, P=2, D=0, Q=3)
            return self.predict(), 1

    def get_err(self, data):
        # mse = model.mse
        mae = sum(
            [abs(data) for data in correct_max_key(self.model.predict(0, data.size - 1)) - data]) / data.size
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        # from statsmodels.tsa.stattools import acf, pacf
        # acf_list = acf(data, nlags=10)
        # pacf_list = pacf(data, nlags=10)
        #       AR(p)         MA(q)            ARMA(p,q)
        # ACF   拖尾           截尾+q阶后为0     拖尾+q阶后为0
        # PACF  截尾+p阶后为0   拖尾             拖尾+p阶后为0
        # plot_acf(data, lags=10).show()
        # plot_pacf(data, lags=10).show()
        data = self.init_data(data)
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
                                    pool.apply_async(self.build, (data, p, d, q, P, D, Q, True))
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

    def __init__(self, lag, predict_step, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'rnn/'

    def init_data(self, data):
        group_num = len(data) - self.lag - self.predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, k)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, k)])
            test_x = np.array([data[i:i + self.lag] for i in range(k, group_num)])
            test_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(k, group_num)])
        else:  # if data is not enough, keep the same between train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, group_num)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, group_num)])
            test_x = train_x
            test_y = train_y
        pre_x = np.expand_dims(np.array(data[-self.lag:]), 0)
        return train_x, train_y, test_x, test_y, pre_x

    def build(self, train_x, train_y, test_x, test_y,
              activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        self.model = Sequential([
            SimpleRNN(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            SimpleRNN(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = self.model.fit(train_x, train_y, validation_data=(test_x, test_y),
                                 epochs=100, batch_size=batch_size,
                                 callbacks=[early_stopping], verbose=0)
        self.err = self.get_err(test_x, test_y)
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, self.err))
            plt.close()

    def predict(self, pre_x):
        return correct_max_key(self.model.predict(pre_x)[0]).tolist()

    def train(self, data):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        self.build(train_x, train_y, test_x, test_y,
                   activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0, learning_rate=0.01,
                   batch_size=4)
        return self.predict(pre_x), 1

    def retrain(self, data, threshold_err):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_x, test_y)
        if self.err <= threshold_err * old_err:
            return self.predict(pre_x), 0
        else:
            self.build(train_x, train_y, test_x, test_y,
                       activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0, learning_rate=0.01,
                       batch_size=4)
            return self.predict(pre_x), 1

    def get_err(self, test_x, test_y):
        # ERROR: loss里的mse和实际计算的mse有差距
        mae = sum(sum([abs(pre - true)
                       for pre, true in
                       zip(correct_max_key(self.model.predict(test_x)), test_y)])) / test_y.size
        # mse = history.history['val_loss'][-1]
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
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
                                                         (train_x, train_y, test_x, test_y,
                                                          activation, unit1, unit2, dropout1, dropout2, learning_rate,
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

    def __init__(self, lag, predict_step, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'lstm/'

    def init_data(self, data):
        group_num = len(data) - self.lag - self.predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, k)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, k)])
            test_x = np.array([data[i:i + self.lag] for i in range(k, group_num)])
            test_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(k, group_num)])
        else:  # if data is not enough, keep the same between train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, group_num)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, group_num)])
            test_x = train_x
            test_y = train_y
        pre_x = np.expand_dims(np.array(data[-self.lag:]), 0)
        return train_x, train_y, test_x, test_y, pre_x

    def build(self, train_x, train_y, test_x, test_y,
              activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        self.model = Sequential([
            LSTM(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            LSTM(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = self.model.fit(train_x, train_y, validation_data=(test_x, test_y),
                                 epochs=100, batch_size=batch_size,
                                 callbacks=[early_stopping], verbose=0)
        self.err = self.get_err(test_x, test_y)
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, self.err))
            plt.close()

    def predict(self, pre_x):
        return correct_max_key(self.model.predict(pre_x)[0]).tolist()

    def train(self, data):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        self.build(train_x, train_y, test_x, test_y,
                   activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0, learning_rate=0.01,
                   batch_size=4)
        return self.predict(pre_x), 1

    def retrain(self, data, threshold_err):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_x, test_y)
        if self.err <= threshold_err * old_err:
            return self.predict(pre_x), 0
        else:
            self.build(train_x, train_y, test_x, test_y,
                       activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0, learning_rate=0.01,
                       batch_size=4)
            return self.predict(pre_x), 1

    def get_err(self, test_x, test_y):
        # ERROR: loss里的mse和实际计算的mse有差距
        mae = sum(sum([abs(pre - true)
                       for pre, true in
                       zip(correct_max_key(self.model.predict(test_x)), test_y)])) / test_y.size
        # mse = history.history['val_loss'][-1]
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
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
                                                         (train_x, train_y, test_x, test_y,
                                                          activation, unit1, unit2, dropout1, dropout2, learning_rate,
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

    def __init__(self, lag, predict_step, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.model_path = model_path + 'gru/'

    def init_data(self, data):
        group_num = len(data) - self.lag - self.predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, k)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, k)])
            test_x = np.array([data[i:i + self.lag] for i in range(k, group_num)])
            test_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(k, group_num)])
        else:  # if data is not enough, keep the same between train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, group_num)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, group_num)])
            test_x = train_x
            test_y = train_y
        pre_x = np.expand_dims(np.array(data[-self.lag:]), 0)
        return train_x, train_y, test_x, test_y, pre_x

    def build(self, train_x, train_y, test_x, test_y,
              activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        start_time = time.time()
        self.model = Sequential([
            GRU(activation=activation, units=unit1, input_shape=(self.lag, 1), return_sequences=True),
            # Dropout(dropout1),
            GRU(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.001, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = self.model.fit(train_x, train_y, validation_data=(test_x, test_y),
                                 epochs=100, batch_size=batch_size,
                                 callbacks=[early_stopping], verbose=0)
        self.err = self.get_err(test_x, test_y)
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size,
                    end_time - start_time, self.err))
            plt.close()

    def predict(self, pre_x):
        return correct_max_key(self.model.predict(pre_x)[0]).tolist()

    def train(self, data):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        self.build(train_x, train_y, test_x, test_y,
                   activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0, learning_rate=0.01,
                   batch_size=4)
        return self.predict(pre_x), 1

    def retrain(self, data, threshold_err):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_x, test_y)
        if self.err <= threshold_err * old_err:
            return self.predict(pre_x), 0
        else:
            self.build(train_x, train_y, test_x, test_y,
                       activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0, learning_rate=0.01,
                       batch_size=4)
            return self.predict(pre_x), 1

    def get_err(self, test_x, test_y):
        # ERROR: loss里的mse和实际计算的mse有差距
        mae = sum(sum([abs(pre - true)
                       for pre, true in
                       zip(correct_max_key(self.model.predict(test_x)), test_y)])) / test_y.size
        # mse = history.history['val_loss'][-1]
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
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
                                                         (train_x, train_y, test_x, test_y,
                                                          activation, unit1, unit2, dropout1, dropout2, learning_rate,
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

    def __init__(self, lag, predict_step, width, model_path):
        super().__init__()
        self.predict_step = predict_step
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'var/'
        self.model = None
        self.err = 0

    def init_data(self, data):
        data = np.array(data)
        k = int(0.7 * len(data))
        if len(data) - k >= self.lag + self.predict_step:  # if data is enough, split into train_data and test_data
            train_data = data[:k]
            test_data = data[k:]
        else:  # if data is not enough, keep the same between train_data and test_data
            train_data = data
            test_data = data
        return train_data, test_data

    def build(self, train_data, test_data, p, is_plot=False):
        try:
            model = VAR(train_data).fit(maxlags=self.lag, verbose=False, trend='c')
        except ValueError:
            # 数据异常，只能放弃趋势
            try:
                self.model = VAR(train_data).fit(maxlags=self.lag, verbose=False, trend='n')
            except LinAlgError:
                self.model = VAR(train_data[1:]).fit(maxlags=self.lag, verbose=False, trend='n')
        self.err = self.get_err(test_data)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(test_data[-1])
            plt.plot(self.model.forecast(test_data[-self.lag:], steps=1))
            plt.savefig(
                self.model_path + "default_%s.png" % self.err)
            plt.close()

    def predict(self, data):
        return correct_cdf(self.model.forecast(data, steps=self.predict_step)).tolist()

    def train(self, data):
        train_data, test_data = self.init_data(data)
        self.build(train_data, test_data, None)
        return self.predict(test_data[-self.lag:]), 1

    def retrain(self, data, threshold_err):
        train_data, test_data = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_data)
        if self.err <= threshold_err * old_err:
            return self.predict(test_data[-self.lag:]), 0
        else:
            self.build(train_data, test_data, None)
            return self.predict(test_data[-self.lag:]), 1

    def get_err(self, data):
        test_group_num = len(data) - self.lag - self.predict_step + 1
        mae = sum([sum([abs(err)
                        for l in (correct_cdf(self.model.forecast(data[i:i + self.lag], steps=self.predict_step))
                                  - data[i + self.lag: i + self.lag + self.predict_step])
                        for err in l])
                   for i in range(test_group_num)]) / (test_group_num * self.width * self.predict_step)
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_data, test_data = self.init_data(data)
        pool = multiprocessing.Pool(processes=thread)
        pool.apply_async(self.build, (train_data, test_data, None, True))
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
    ERROR: data.shape是[31,100]时报错，但[31,2]时不报错
    grid search:
    """

    def __init__(self, lag, predict_step, width, model_path):
        super().__init__()
        self.predict_step = predict_step
        self.lag = lag
        self.width = width
        self.model_path = model_path + 'vsarima/'
        self.model = None
        self.err = 0

    def init_data(self, data):
        data = np.array(data)
        k = int(0.7 * len(data))
        if len(data) - k >= self.lag + self.predict_step:  # if data is enough, split into train_data and test_data
            train_data = data[:k]
            test_data = data[k:]
        else:  # if data is not enough, keep the same between train_data and test_data
            train_data = data
            test_data = data
        return train_data, test_data

    def build(self, train_data, test_data, p, q, is_plot=False):
        self.model = VARMAX(train_data, order=(p, q),
                            error_cov_type='error_cov_type',
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
        self.err = self.get_err(test_data)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(test_data[-1])
            plt.plot(self.model.forecast(test_data[-self.lag:], steps=1))
            plt.savefig(
                self.model_path + "%s_%s_%s.png" % (p, q, self.err))
            plt.close()

    def predict(self, data):
        return correct_cdf(self.model.forecast(data, steps=self.predict_step)).tolist()

    def train(self, data):
        train_data, test_data = self.init_data(data)
        self.build(train_data, test_data, p=2, q=0)
        return self.predict(test_data[-self.lag:]), 1

    def retrain(self, data, threshold_err):
        train_data, test_data = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_data)
        if self.err <= threshold_err * old_err:
            return self.predict(test_data[-self.lag:]), 0
        else:
            self.build(train_data, test_data, p=2, q=0)
            return self.predict(test_data[-self.lag:]), 1

    def get_err(self, data):
        test_group_num = len(data) - self.lag - self.predict_step + 1
        mae = sum([sum([abs(err)
                        for l in (correct_cdf(self.model.forecast(data[i:i + self.lag], steps=self.predict_step))
                                  - data[i + self.lag: i + self.lag + self.predict_step])
                        for err in l])
                   for i in range(test_group_num)]) / (test_group_num * self.width * self.predict_step)
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_data, test_data = self.init_data(data)
        ps = qs = [0, 1, 2, 3]
        pool = multiprocessing.Pool(processes=thread)
        i = 0
        for p in ps:
            for q in qs:
                i += 1
                if i > start_num:
                    pool.apply_async(self.build, (train_data, test_data, p, q, True))
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

    def __init__(self, lag, predict_step, width, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.width = width
        self.model_path = model_path + 'fclstm/'
        self.model = None
        self.err = 0

    def init_data(self, data):
        group_num = len(data) - self.lag - self.predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, k)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, k)])
            train_y = train_y.reshape(train_y.shape[0], self.predict_step * self.width)
            test_x = np.array([data[i:i + self.lag] for i in range(k, group_num)])
            test_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(k, group_num)])
            test_y = test_y.reshape(test_y.shape[0], self.predict_step * self.width)
        else:  # if data is not enough, keep the same between train_data and test_data
            train_x = np.array([data[i:i + self.lag] for i in range(0, group_num)])
            train_y = np.array([data[i + self.lag:i + self.lag + self.predict_step] for i in range(0, group_num)])
            train_y = train_y.reshape(train_y.shape[0], self.predict_step * self.width)
            test_x = train_x
            test_y = train_y
        pre_x = np.expand_dims(np.array(data[-self.lag:]), 0)
        return train_x, train_y, test_x, test_y, pre_x

    def build(self, train_x, train_y, test_x, test_y,
              activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, is_plot=False):
        self.model = Sequential([
            LSTM(activation=activation, units=unit1, input_shape=(self.lag, self.width), return_sequences=True),
            # Dropout(dropout1),
            LSTM(activation=activation, units=unit2, return_sequences=False),
            # Dropout(dropout2),
            Dense(units=self.predict_step * self.width)
        ])
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.0005, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = self.model.fit(train_x, train_y, validation_data=(test_x, test_y),
                                 epochs=100, batch_size=batch_size,
                                 callbacks=[early_stopping], verbose=0)
        self.err = self.get_err(test_x, test_y)
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation, unit1, unit2, dropout1, dropout2, learning_rate, batch_size, self.err))
            plt.close()

    def predict(self, data):
        return correct_cdf(self.model.predict(data).reshape(self.predict_step, self.width)).tolist()

    def train(self, data):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        self.build(train_x, train_y, test_x, test_y,
                   activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0,
                   learning_rate=0.01, batch_size=4)
        return self.predict(pre_x), 1

    def retrain(self, data, threshold_err):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_x, test_y)
        if self.err <= threshold_err * old_err:
            return self.predict(pre_x), 1
        else:
            self.build(train_x, train_y, test_x, test_y,
                       activation='relu', unit1=128, unit2=256, dropout1=0.0, dropout2=0.0,
                       learning_rate=0.01, batch_size=4)
            return self.predict(pre_x), 1

    def get_err(self, test_x, test_y):
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = history.history['val_loss'][-1]
        pres = correct_cdf(self.model.predict(test_x).reshape(test_x.shape[0] * self.predict_step, self.width))
        trues = test_y.reshape(test_x.shape[0] * self.predict_step, self.width)
        mae = np.sum(np.abs(pres - trues)) / test_y.size
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
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
                                                         (train_x, train_y, test_x, test_y,
                                                          activation, unit1, unit2, dropout1, dropout2, learning_rate,
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

    def __init__(self, lag, predict_step, width, model_path):
        super().__init__()
        self.lag = lag
        self.predict_step = predict_step
        self.width = width
        self.model_path = model_path + 'convlstm/'
        self.model = None
        self.err = 0

    def init_data(self, data):
        group_num = len(data) - self.lag - self.predict_step + 1
        k = int(0.7 * group_num)
        if k:  # if data is enough, split into train_data and test_data
            train_x = np.expand_dims(np.array([data[i:i + self.lag]
                                               for i in range(0, k)]), -1)
            train_y = np.expand_dims(np.array([data[i + self.lag:i + self.lag + self.predict_step]
                                               for i in range(0, k)]), -1)
            test_x = np.expand_dims(np.array([data[i:i + self.lag]
                                              for i in range(k, group_num)]), -1)
            test_y = np.expand_dims(np.array([data[i + self.lag:i + self.lag + self.predict_step]
                                              for i in range(k, group_num)]), -1)
        else:  # if data is not enough, keep the same between train_data and test_data
            train_x = np.expand_dims(np.array([data[i:i + self.lag]
                                               for i in range(0, group_num)]), -1)
            train_y = np.expand_dims(np.array([data[i + self.lag:i + self.lag + self.predict_step]
                                               for i in range(0, group_num)]), -1)
            test_x = train_x
            test_y = train_y
        pre_x = np.expand_dims(np.array(data[-self.lag:]), 0)
        return train_x, train_y, test_x, test_y, pre_x

    def build(self, train_x, train_y, test_x, test_y,
              activation1, activation2, filter1, filter2, dropout1, dropout2, kernal_size,
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
        # self.model = Sequential([
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

        # 2. ConvLSTM编码-Reshape+Conv2D解码
        self.model = Sequential([
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
        self.model.compile(optimizer=optimizer, loss='mse')
        early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.0005, verbose=0)
        # reduce_lr = ReduceLROnPlateau(monitor="loss", patience=5, verbose=0)
        history = self.model.fit(train_x, train_y, validation_data=(test_x, test_y),
                                 epochs=100, batch_size=batch_size,
                                 callbacks=[early_stopping], verbose=0)
        self.err = self.get_err(test_x, test_y)
        end_time = time.time()
        if is_plot:
            if os.path.exists(self.model_path) is False:
                os.makedirs(self.model_path)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.savefig(
                self.model_path + "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.png" % (
                    activation1, activation2, filter1, filter2, dropout1, dropout2,
                    kernal_size, learning_rate, batch_size, end_time - start_time, self.err))
            plt.close()

    def predict(self, data):
        return correct_cdf(self.model.predict(data)[0, :, :, 0]).tolist()

    def train(self, data):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        self.build(train_x, train_y, test_x, test_y, activation1='tanh', activation2='tanh',
                   filter1=8, filter2=8, dropout1=0.0, dropout2=0.0, kernal_size=9,
                   learning_rate=0.01, batch_size=32)
        return self.predict(pre_x), 1

    def retrain(self, data, threshold_err):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
        old_err = self.err
        self.err = self.get_err(test_x, test_y)
        if self.err <= threshold_err * old_err:
            return self.predict(pre_x), 1
        else:
            self.build(train_x, train_y, test_x, test_y, activation1='tanh', activation2='tanh',
                       filter1=8, filter2=8, dropout1=0.0, dropout2=0.0, kernal_size=9,
                       learning_rate=0.01, batch_size=32)
            return self.predict(pre_x), 1

    def get_err(self, test_x, test_y):
        # ERROR: loss里的mse和实际计算的mse有差距
        # mse = history.history['val_loss'][-1]
        pres = self.model.predict(test_x).reshape(test_x.shape[0] * self.predict_step, self.width)
        trues = test_y.reshape(test_x.shape[0] * self.predict_step, self.width)
        mae = np.sum(np.abs(pres - trues)) / test_y.size
        return mae

    def grid_search(self, data, thread=1, start_num=0):
        train_x, train_y, test_x, test_y, pre_x = self.init_data(data)
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
                                                                 (train_x, train_y, test_x, test_y,
                                                                  activation1, activation2, filter1, filter2,
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
