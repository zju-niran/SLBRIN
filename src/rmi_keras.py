import logging
import os.path
import random
import shutil
from functools import wraps

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')  # 解决_tkinter.TclError: couldn't connect to display "localhost:11.0"
import matplotlib.pyplot as plt

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

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def elu(x, alpha=1):
        a = x[x > 0]
        b = alpha * (np.exp(x[x < 0]) - 1)
        result = np.concatenate((b, a), axis=0)
        return result

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # @memoize TODO: 要加缓存的话， 缓存的key不能是list，之前是float
    # TODO: 和model.predict有小偏差，怀疑是exp的e和elu的e不一致
    def predict(self, input_keys):
        input_keys = nparray_normalize_minmax(input_keys, self.input_min, self.input_max)
        tmp_res = np.mat(input_keys).T
        for i in range(len(self.core_nums) - 1):
            # w * x + b
            # sigmoid(x)
            tmp_res = AbstractNN.sigmoid(tmp_res * np.mat(self.weights[i * 2]) + np.mat(self.weights[i * 2 + 1]))
        # clip到最大最小值之间
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
    def __init__(self, model_path, model_index, train_x, train_y, threshold, use_threshold, cores, train_step_num,
                 batch_size, learning_rate, retrain_time_limit):
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_x, self.train_x_min, self.train_x_max = nparray_normalize(train_x)
        self.train_y, self.train_y_min, self.train_y_max = nparray_normalize(train_y)
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.model = None
        self.min_err, self.max_err = 0, 0
        self.retrain_times = 0
        self.retrain_time_limit = retrain_time_limit
        self.model_path = model_path
        self.model_index = model_index
        self.model_hdf_dir = model_path + "hdf/"
        self.model_png_dir = model_path + "png/"
        self.model_loss_dir = model_path + "loss/"
        if os.path.exists(self.model_hdf_dir) is False:
            os.makedirs(self.model_hdf_dir)
        if os.path.exists(self.model_png_dir) is False:
            os.makedirs(self.model_png_dir)
        if os.path.exists(self.model_loss_dir) is False:
            os.makedirs(self.model_loss_dir)
        self.model_hdf_file = os.path.join(self.model_hdf_dir, self.model_index + "_weights.best.hdf5")
        self.model_png_file = os.path.join(self.model_png_dir, self.model_index + "_weights.best.png")
        self.model_loss_file = os.path.join(self.model_loss_dir, self.model_index + "_weights.best.loss")
        self.clean_not_best_model_file()
        self.get_best_model_file()

    # train model
    def train(self):
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
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
        if self.is_model_file_valid() is False:  # valid the model file exists and is in hdf5 format
            # delete model when model file is not in hdf5 format but exists, maybe destroyed by interrupt
            if os.path.exists(self.model_hdf_file):
                os.remove(self.model_hdf_file)
            model = tf.keras.Sequential()
            for i in range(len(self.core_nums) - 2):
                model.add(tf.keras.layers.Dense(units=self.core_nums[i + 1],
                                                input_dim=self.core_nums[i],
                                                activation='sigmoid'))
            model.add(tf.keras.layers.Dense(units=self.core_nums[-1],
                                            activation='sigmoid'))
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            model.compile(optimizer=optimizer, loss=self.score)
            self.model = model
            # self.model.summary()
            checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_hdf_file,
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
                                                          factor=0.5,
                                                          patience=450,
                                                          verbose=0,
                                                          mode='min',
                                                          min_lr=0.0001)
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.model_loss_file,
                                                         histogram_freq=1)
            callbacks_list = [checkpoint, reduce, early_stopping, tensorboard]
            # fit and save model
            history = self.model.fit(self.train_x, self.train_y,
                                     epochs=self.train_step_nums,
                                     initial_epoch=0,
                                     batch_size=self.batch_size,
                                     verbose=2,
                                     callbacks=callbacks_list)
        self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
        self.min_err, self.max_err = self.get_err()
        err_length = self.max_err - self.min_err
        self.rename_model_file_by_err(err_length)
        if self.use_threshold:
            if err_length > self.threshold:
                if self.retrain_times < self.retrain_time_limit:
                    self.init_model_name_by_random()
                    self.retrain_times += 1
                    print("Retrain %d when score not perfect: Model %s, Err %f, Threshold %f" % (
                        self.retrain_times, self.model_hdf_file, err_length, self.threshold))
                    logging.info("Retrain %d when score not perfect: Model %s, Err %f, Threshold %f" % (
                        self.retrain_times, self.model_hdf_file, err_length, self.threshold))
                    self.train()
                else:
                    print("Retrain time limit: Model %s, Err %f, Threshold %f" % (
                        self.model_hdf_file, err_length, self.threshold))
                    logging.info("Retrain time limit: Model %s, Err %f, Threshold %f" % (
                        self.model_hdf_file, err_length, self.threshold))
                    self.get_best_model_file()
                    self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
                    self.min_err, self.max_err = self.get_err()
                    return
            else:
                print("Model perfect: Model %s, Err %f, Threshold %f" % (
                    self.model_hdf_file, err_length, self.threshold))
                logging.info("Model perfect: Model %s, Err %f, Threshold %f" % (
                    self.model_hdf_file, err_length, self.threshold))
        else:
            print("Stop train when train early stop or epoch finish: Model %s, Err %f, Threshold %f" % (
                self.model_hdf_file, err_length, self.threshold))
            logging.info("Stop train when train early stop or epoch finish: Model %s, Err %f, Threshold %f" % (
                self.model_hdf_file, err_length, self.threshold))

    def is_model_file_valid(self):
        try:
            model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
            return True
        except Exception:
            # {ValueError}No model config found in the file
            # {OSError}SavedModel file does not exist
            return False

    def get_weights(self):
        return self.model.get_weights()

    def score(self, y_true, y_pred):
        # 这里的y应该是局部的，因此scores和err算出来不一致
        y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
        diff_clip = y_true - y_pred_clip
        range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
        diff = y_true - y_pred
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return 0.1 * range_loss + mse_loss

    def get_err(self):
        pres = self.model.predict(self.train_x).flatten()
        errs_normalize_reverse = nparray_diff_normalize_reverse(pres, self.train_y, self.train_y_min, self.train_y_max)
        return errs_normalize_reverse.min(), errs_normalize_reverse.max()

    def predict(self):
        pres = self.model.predict(self.train_x).flatten()
        return nparray_normalize_reverse(pres, self.train_y_min, self.train_y_max)

    def plot(self):
        pres = self.model.predict(self.train_x).flatten()
        plt.plot(self.train_x, self.train_y, 'y--', label="true")
        plt.plot(self.train_x, pres, 'm--', label="predict")
        plt.legend()
        plt.savefig(self.model_png_file)
        plt.close()

    def get_best_model_file(self):
        """
        find the min err model path
        """
        min_err = 'best'
        files = os.listdir(self.model_hdf_dir)
        for file in files:
            tmp_split = file.split('.')
            tmp_model_index = tmp_split[0]
            if tmp_model_index == self.model_index:
                tmp_err = '.'.join(tmp_split[1:-1])
                try:
                    tmp_err = float(tmp_err)
                except:
                    continue
                if (min_err == 'best' or min_err > tmp_err) and tmp_err >= 0:
                    min_err = tmp_err
        best_file_name = '.'.join([self.model_index, str(min_err), "hdf5"])
        best_png_name = '.'.join([self.model_index, str(min_err), "png"])
        best_loss_name = '.'.join([self.model_index, str(min_err), "loss"])
        self.model_hdf_file = os.path.join(self.model_hdf_dir, best_file_name)
        self.model_png_file = os.path.join(self.model_png_dir, best_png_name)
        self.model_loss_file = os.path.join(self.model_loss_dir, best_loss_name)

    def rename_model_file_by_err(self, err):
        """
        rename model file by err
        :param err: float
        """
        new_file_name = '.'.join([self.model_index, str(err), "hdf5"])
        new_png_name = '.'.join([self.model_index, str(err), "png"])
        new_loss_name = '.'.join([self.model_index, str(err), "loss"])
        new_model_path = os.path.join(self.model_hdf_dir, new_file_name)
        new_png_path = os.path.join(self.model_png_dir, new_png_name)
        new_loss_path = os.path.join(self.model_loss_dir, new_loss_name)
        try:
            os.rename(self.model_hdf_file, new_model_path)
        except FileExistsError:  # 相同误差的model不再重复保存
            os.remove(self.model_hdf_file)
        except FileNotFoundError:  # 原模型不存在则跳过
            pass
        try:
            os.rename(self.model_png_file, new_png_path)
        except FileExistsError:
            os.remove(self.model_png_file)
        except FileNotFoundError:
            pass
        try:
            os.rename(self.model_loss_file, new_loss_path)
        except FileExistsError:
            shutil.rmtree(self.model_loss_file)
        except FileNotFoundError:
            pass
        self.model_hdf_file = new_model_path
        self.model_png_file = new_png_path
        self.model_loss_file = new_loss_path

    def init_model_name_by_random(self):
        """
        init model name by random float num
        """
        random_str = str(random.random() * -1)
        new_file_name = '.'.join([self.model_index, random_str, "hdf5"])
        new_png_name = '.'.join([self.model_index, random_str, "png"])
        new_loss_name = '.'.join([self.model_index, random_str, "loss"])
        self.model_hdf_file = os.path.join(self.model_hdf_dir, new_file_name)
        self.model_png_file = os.path.join(self.model_png_dir, new_png_name)
        self.model_loss_file = os.path.join(self.model_loss_dir, new_loss_name)

    def clean_not_best_model_file(self):
        """
        delete all the model file besides the best one
        """
        min_err = 'best'
        files = os.listdir(self.model_hdf_dir)
        for file in files:
            tmp_split = file.split('.')
            tmp_model_index = tmp_split[0]
            if tmp_model_index == self.model_index:
                tmp_err = '.'.join(tmp_split[1:-1])
                try:
                    tmp_err = float(tmp_err)
                except:
                    continue
                if (min_err == 'best' or min_err > tmp_err) and tmp_err >= 0:
                    tmp_min_err = tmp_err
                    tmp_max_err = min_err
                else:
                    tmp_min_err = min_err
                    tmp_max_err = tmp_err
                last_min_file_name = '.'.join([self.model_index, str(tmp_max_err), "hdf5"])
                last_min_png_name = '.'.join([self.model_index, str(tmp_max_err), "png"])
                last_min_loss_name = '.'.join([self.model_index, str(tmp_max_err), "loss"])
                if os.path.exists(os.path.join(self.model_hdf_dir, last_min_file_name)):
                    os.remove(os.path.join(self.model_hdf_dir, last_min_file_name))
                if os.path.exists(os.path.join(self.model_png_dir, last_min_png_name)):
                    os.remove(os.path.join(self.model_png_dir, last_min_png_name))
                if os.path.exists(os.path.join(self.model_loss_dir, last_min_loss_name)):
                    shutil.rmtree(os.path.join(self.model_loss_dir, last_min_loss_name))
                min_err = tmp_min_err
