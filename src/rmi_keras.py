import logging
import os.path
import random
from functools import wraps

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.spatial_index.common_utils import nparray_normalize, nparray_normalize_minmax, nparray_diff_normalize_reverse, \
    nparray_normalize_reverse


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
    def __init__(self, weights, core_nums, input_min, input_max, output_min, output_max, min_err, max_err):
        self.weights = weights
        self.core_nums = core_nums
        self.input_min = input_min
        self.input_max = input_max
        self.output_min = output_min
        self.output_max = output_max
        self.min_err = min_err
        self.max_err = max_err

    # @memoize TODO: 要加缓存的话， 缓存的key不能是list，之前是float
    # TODO: 和model.predict有小偏差，怀疑是exp的e和elu的e不一致
    def predict(self, input_keys):
        input_keys = nparray_normalize_minmax(input_keys, self.input_min, self.input_max)
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
        # 最后一层单独用relu，值clip到最大最小值之间
        tmp_res = tmp_res * np.mat(self.weights[-2]) + np.mat(self.weights[-1])
        tmp_res = np.asarray(tmp_res).flatten()
        return nparray_normalize_reverse(tmp_res, self.output_min, self.output_max)


    @staticmethod
    def init_by_dict(d: dict):
        return AbstractNN(d['weights'], d['core_nums'],
                          d['input_min'], d['input_max'],
                          d['output_min'], d['output_max'],
                          d['min_err'], d['max_err'], )


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
        self.train_x, self.train_x_min, self.train_x_max = nparray_normalize(train_x)
        self.clean_not_best_model_file(model_path)
        self.train_y, self.train_y_min, self.train_y_max = nparray_normalize(train_y)
        self.model_path = self.get_best_model_file(model_path)
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.model = None
        self.min_err, self.max_err = 0, 0
        self.retrain_times = 0
        self.retrain_time_limit = 1

    # train model
    def train(self):
        model_dir = os.path.join(os.path.dirname(self.model_path))
        if os.path.exists(model_dir) is False:
            os.makedirs(model_dir)
        logging.basicConfig(filename=os.path.join(model_dir, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S %p")
        # GPU配置
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # 不输出报错：This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the
        # following CPU instructions in performance-critical operations:  AVX AVX2
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        if self.is_model_file_valid():  # valid the model file exists and is in hdf5 format
            self.model = tf.keras.models.load_model(self.model_path, custom_objects={'score': self.score})
            # do not train exists model when err is enough
            if self.use_threshold:
                self.min_err, self.max_err = self.get_err()
                err_length = self.max_err - self.min_err
                if err_length <= self.threshold:
                    print("Do not train when model exists and prefect: Model %s, Err %f, Threshold %f" % (
                        self.model_path, err_length, self.threshold))
                    logging.info("Do not train when model exists and prefect: Model %s, Err %f, Threshold %f" % (
                        self.model_path, err_length, self.threshold))
                    self.model_path = self.rename_model_file_by_err(self.model_path, err_length)
                    return
        else:
            # delete model when model file is not in hdf5 format but exists, maybe destroyed by interrupt
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
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

            model.compile(optimizer=optimizer, loss=self.score)
            self.model = model
        # self.model.summary()
        # checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_path,
                                                        monitor='loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        mode='min',
                                                        save_freq='epoch')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=500,
                                                          mode='min',
                                                          verbose=0)
        reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                      factor=0.90,
                                                      patience=400,
                                                      verbose=0,
                                                      mode='min',
                                                      min_lr=0.0001)
        callbacks_list = [checkpoint, reduce, early_stopping]
        # fit and save model
        history = self.model.fit(self.train_x, self.train_y,
                                 epochs=self.train_step_nums,
                                 initial_epoch=0,
                                 batch_size=self.batch_size,
                                 verbose=0,
                                 callbacks=callbacks_list)
        self.model = tf.keras.models.load_model(self.model_path, custom_objects={'score': self.score})
        self.min_err, self.max_err = self.get_err()
        err_length = self.max_err - self.min_err
        self.model_path = self.rename_model_file_by_err(self.model_path, err_length)
        if self.use_threshold:
            if err_length > self.threshold:
                if self.retrain_times < self.retrain_time_limit:
                    self.model_path = self.init_model_name_by_random(self.model_path)
                    self.retrain_times += 1
                    print("Retrain %d when score not perfect: Model %s, Err %f, Threshold %f" % (
                        self.retrain_times, self.model_path, err_length, self.threshold))
                    logging.info("Retrain %d when score not perfect: Model %s, Err %f, Threshold %f" % (
                        self.retrain_times, self.model_path, err_length, self.threshold))
                    self.train()
                else:
                    print("Retrain time limit: Model %s, Err %f, Threshold %f" % (
                        self.model_path, err_length, self.threshold))
                    logging.info("Retrain time limit: Model %s, Err %f, Threshold %f" % (
                        self.model_path, err_length, self.threshold))
                    self.model_path = self.get_best_model_file(self.model_path)
                    self.model = tf.keras.models.load_model(self.model_path, custom_objects={'score': self.score})
                    self.min_err, self.max_err = self.get_err()
                    return
            else:
                print("Model perfect: Model %s, Err %f, Threshold %f" % (
                    self.model_path, err_length, self.threshold))
                logging.info("Model perfect: Model %s, Err %f, Threshold %f" % (
                    self.model_path, err_length, self.threshold))
        else:
            print("Stop train when score stop decreasing: Model %s, Err %f, Threshold %f" % (
                self.model_path, err_length, self.threshold))
            logging.info("Stop train when score stop decreasing: Model %s, Err %f, Threshold %f" % (
                self.model_path, err_length, self.threshold))

    def is_model_file_valid(self):
        try:
            model = tf.keras.models.load_model(self.model_path, custom_objects={'score': self.score})
            return True
        except Exception:
            # {ValueError}No model config found in the file
            # {OSError}SavedModel file does not exist
            return False

    def get_weights(self):
        return self.model.get_weights()

    def score(self, y_true, y_pred):
        # 这里的y应该是局部的，因此scores和err算出来不一致
        diff = y_true - y_pred
        range_loss = tf.keras.backend.max(diff) - tf.keras.backend.min(diff)
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return 2 * range_loss + mse_loss

    def get_err(self):
        pres = self.model.predict(self.train_x).flatten()
        errs_normalize_reverse = nparray_diff_normalize_reverse(pres, self.train_y, self.train_y_min, self.train_y_max)
        return errs_normalize_reverse.min(), errs_normalize_reverse.max()

    def plot(self):
        pres = self.model.predict(self.train_x).flatten()
        plt.plot(self.train_x, self.train_y, 'y--', label="true")
        plt.plot(self.train_x, pres, 'm--', label="predict")
        png_path = self.model_path.replace("hdf5", "png")
        png_path = png_path.replace("models", "models_png")
        file_path, file_name = os.path.split(png_path)
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        plt.legend()
        plt.savefig(png_path)
        plt.close()

    @staticmethod
    def get_best_model_file(model_path):
        """
        find the min err model path
        :return: perfect model path
        """
        file_path, file_name = os.path.split(model_path)
        tmp_split = file_name.split('.')
        model_index = tmp_split[0]
        suffix = tmp_split[-1]
        min_err = 'best'
        if os.path.exists(file_path) is False:
            return model_path
        files = os.listdir(file_path)
        for file in files:
            tmp_split = file.split('.')
            tmp_model_index = tmp_split[0]
            if tmp_model_index == model_index:
                tmp_err = '.'.join(tmp_split[1:-1])
                try:
                    tmp_err = float(tmp_err)
                except:
                    continue
                if (min_err == 'best' or min_err > tmp_err) and tmp_err >= 0:
                    min_err = tmp_err
        best_file_name = '.'.join([model_index, str(min_err), suffix])
        return os.path.join(file_path, best_file_name)

    @staticmethod
    def rename_model_file_by_err(model_path, err):
        """
        rename model file by err
        :param model_path: old path
        :param err: float
        :return: new path
        """
        file_path, file_name = os.path.split(model_path)
        tmp_split = file_name.split('.')
        model_index = tmp_split[0]
        suffix = tmp_split[-1]
        new_file_name = '.'.join([model_index, str(err), suffix])
        try:
            new_model_path = os.path.join(file_path, new_file_name)
            os.rename(model_path, new_model_path)
            return new_model_path
        except FileExistsError:  # 相同误差的model不再重复保存
            os.remove(model_path)
            return model_path

    @staticmethod
    def init_model_name_by_random(model_path):
        """
        init model name by random float num
        :param model_path: old path
        :return: new path
        """
        file_path, file_name = os.path.split(model_path)
        tmp_split = file_name.split('.')
        model_index = tmp_split[0]
        suffix = tmp_split[-1]
        new_file_name = '.'.join([model_index, str(random.random() * -1), suffix])
        return os.path.join(file_path, new_file_name)

    @staticmethod
    def clean_not_best_model_file(model_path):
        """
        delete all the model file besides the best one
        :return: None
        """
        file_path, file_name = os.path.split(model_path)
        tmp_split = file_name.split('.')
        model_index = tmp_split[0]
        suffix = tmp_split[-1]
        min_err = 'best'
        if os.path.exists(file_path) is False:
            return
        files = os.listdir(file_path)
        for file in files:
            tmp_split = file.split('.')
            tmp_model_index = tmp_split[0]
            if tmp_model_index == model_index:
                tmp_err = '.'.join(tmp_split[1:-1])
                try:
                    tmp_err = float(tmp_err)
                except:
                    continue
                if (min_err == 'best' or min_err > tmp_err) and tmp_err >= 0:
                    last_min_file_name = '.'.join([model_index, str(min_err), suffix])
                    if os.path.exists(os.path.join(file_path, last_min_file_name)):
                        os.remove(os.path.join(file_path, last_min_file_name))
                    min_err = tmp_err
                else:
                    os.remove(os.path.join(file_path, file))
