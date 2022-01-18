# Main file for NN model
import os.path
from functools import wraps

import numpy as np
import tensorflow as tf


# using cache
def memoize(func):
    memo = {}

    @wraps(func)
    def wrapper(*args):

        if args in memo:
            return memo[args]
        else:
            rv = func(*args)
            memo[args] = rv
            return rv

    return wrapper


# extract matrix for predicting position
class AbstractNN:
    def __init__(self, weights, core_nums, err, threshold):
        self.weights = weights
        self.core_nums = core_nums
        self.err = err
        self.threshold = threshold

    # @memoize TODO: 要加缓存的话， 缓存的key不能是list，之前是float
    def predict(self, input_keys):
        tmp_res = np.mat(input_keys).T
        for i in range(len(self.core_nums) - 2):
            # w * x + b
            tmp_res = tmp_res * np.mat(self.weights[i * 2]) + np.mat(self.weights[i * 2 + 1])
            # relu
            # tmp_res[tmp_res < 0] = 0
            # elu： 大于0保持，小于0的话计算(exp(x)-1)*alpha，alpha默认=1.0
            for k in range(tmp_res.shape[0]):
                for j in range(tmp_res.shape[1]):
                    if tmp_res[k, j] < 0:
                        tmp_res[k, j] = np.exp(tmp_res[k, j]) - 1
            # batch_normalization: 实现有点问题，bn层四个参数gamma/bita/mean/var
            # https://zhuanlan.zhihu.com/p/100672008
            # 前馈的时候计算y只需要gamma和bita，大致为先normal归一化，然后计算bn=gamma*y+beta
            # TODO: gamma和beta都是shape(8, 1)，gamma*y的话，那y得是shape(1,)
            # x_mean = tmp_res.mean(axis=0)
            # x_var = tmp_res.var(axis=0)
            # x_normalized = (tmp_res - x_mean) / np.sqrt(x_var + 0.001)  # 归一化
            # for k in range(tmp_res.shape[0]):
            #     for j in range(tmp_res.shape[1]):
            #         tmp_res[k, j] = np.mat(self.weights[i * 6 + 2])[k, j] * x_normalized[k, j] + \
            #                         np.mat(self.weights[i * 6 + 3])[k, j]  # 计算bn
        # 最后一层单独用relu
        tmp_res = tmp_res * np.mat(self.weights[(i + 1) * 2]) + np.mat(self.weights[(i + 1) * 2 + 1])
        tmp_res[tmp_res < 0] = 0
        return np.asarray(tmp_res).flatten()

    @staticmethod
    def init_by_dict(d: dict):
        return AbstractNN(d['weights'], d['core_nums'], d['err'], d['threshold'])


# Neural Network Model
class TrainedNN:
    def __init__(self, model_path, train_x, train_y, threshold, use_threshold, cores, train_step_num, batch_size,
                 learning_rate, keep_ratio):
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_ratio = keep_ratio
        self.train_x = train_x
        self.train_y = train_y
        self.model_path = model_path
        self.use_threshold = use_threshold
        # 根据label范围和误差百分比计算误差范围
        # 因为block_size=100，所以index最小间隔是0.01，0.005在四舍五入的时候是最小单位，可以避免train_y长度是0的情况
        self.threshold = max(0.005, threshold * (max(self.train_y) - min(self.train_y)))
        self.model = None
        self.best_model = None
        self.err = 0

    # train model
    def train(self):
        # GPU配置
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            # 动态分配GPU内存
            tf.config.experimental.set_memory_growth(gpu, True)
            # 动态分配GPU内存
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            # )

        # create or load model
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.best_model = self.model
            # do not train exists model when err is enough
            if self.use_threshold:
                self.err = max(self.get_err())
                if self.err <= self.threshold:
                    return
        else:
            model_dir = os.path.dirname(self.model_path)
            if os.path.exists(model_dir) is False:
                os.makedirs(model_dir)
            model = tf.keras.Sequential()
            for i in range(len(self.core_nums) - 2):
                model.add(tf.keras.layers.Dense(units=self.core_nums[i + 1],
                                                input_dim=self.core_nums[i],
                                                activation='elu',
                                                kernel_initializer='he_normal',
                                                bias_initializer='zeros'))  # 使用elu和he_normal的组合避免梯度消失
                # drop_rate = 1 - self.keep_ratio
                # if drop_rate > 0:
                #     model.add(tf.keras.layers.Dropout(rate=drop_rate))  # dropout防止过拟合
                # model.add(tf.keras.layers.BatchNormalization())  #bn可以杜绝梯度消失，但是训练的慢，而且predict我没实现。。。
            model.add(tf.keras.layers.Dense(units=self.core_nums[-1],
                                            activation='relu'))  # 最后一层单独用relu把负数弄到0
            # compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                 clipvalue=1.0)  # clipvalue使用梯度裁剪避免梯度爆炸
            model.compile(optimizer=optimizer, loss='mse', metrics=["accuracy"])
            self.model = model
        # self.model.summary()
        # checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                                        monitor='loss',
                                                        verbose=2,
                                                        save_best_only=True,
                                                        mode='min',
                                                        save_freq='epoch')
        callbacks_list = [checkpoint]
        # fit and save model
        check_step = 100
        check_iter = int(self.train_step_nums / check_step)
        last_min_err = 0
        for current_check_iter in range(check_iter):
            history = self.model.fit(self.train_x, self.train_y,
                                     epochs=check_step + current_check_iter * check_step,
                                     initial_epoch=current_check_iter * check_step,
                                     batch_size=self.batch_size,
                                     verbose=2,
                                     callbacks=callbacks_list)
            self.best_model = tf.keras.models.load_model(self.model_path)
            self.err = max(self.get_err())
            print("Err %f, Threshold %f" % (self.err, self.threshold))
            min_loss = min(history.history.get("loss"))
            if current_check_iter == 0:
                last_min_err = min_loss
            # redo or stop train when loss stop decreasing
            if current_check_iter > 0 and min_loss >= last_min_err:
                # retrain when loss stop decreasing and err exceed the threshold
                if self.err > self.threshold:
                    os.remove(self.model_path)
                    self.train()
                    break
                # stop train early when loss stop decreasing and err is enough
                else:
                    return
            # continue train when loss decrease
            else:
                last_min_err = min_loss
                # use threshold to stop train early
                if self.use_threshold and self.err <= self.threshold:
                    return

    # get weight matrix
    def get_weights(self):
        # weights = self.model.get_weights()  # 最后一次fit的weights
        weights = self.best_model.get_weights()  # loss最低的weights
        w_list = []
        for i in range(len(weights)):
            w_list.append(weights[i].tolist())
        return w_list

    # get err = pre - train_y
    def get_err(self):
        # pres = self.model.predict(self.train_x).flatten()  # 最后一次fit的model
        pres = self.best_model.predict(self.train_x).flatten()  # loss最低的model
        errs = (pres - self.train_y).tolist()
        min_err, max_err = 0, 0
        for err in errs:
            if err < 0:
                if err < min_err:
                    min_err = err
            else:
                if err > max_err:
                    max_err = err
        return [abs(min_err), max_err]
