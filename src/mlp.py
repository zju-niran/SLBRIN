import logging
import math
import os.path
import random
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class MLP:
    """
    基于MLP的学习模型（Learned Model）：Keras实现
    """

    def __init__(self, model_path, model_key, train_x, train_x_min, train_x_max, train_y, train_y_min, train_y_max,
                 is_new, weight, core, train_step, batch_size, learning_rate,
                 use_threshold, threshold, retrain_time_limit):
        # common
        self.name = "MLP"
        self.model_path = model_path
        self.model_hdf_dir = os.path.join(model_path, "hdf/")
        self.model_key = model_key
        # data
        self.train_x = train_x
        self.train_x_min = train_x_min
        self.train_x_max = train_x_max
        self.train_y = train_y
        self.train_y_min = train_y_min
        self.train_y_max = train_y_max
        # model structure
        self.is_new = is_new
        self.weight = weight
        self.core = core
        self.train_step = train_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.retrain_times = 0
        self.retrain_time_limit = retrain_time_limit
        # for compute
        self.model = None
        self.matrices = None
        self.min_err = None
        self.max_err = None
        self.logging = None

    def build(self):
        if self.is_new:
            self.init()
            self.model_hdf_file = os.path.join(self.model_hdf_dir, '.'.join([self.model_key, "best", "hdf5"]))
        else:
            self.get_best_model_file()
            try:
                self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.mse})
            except Exception:
                # {ValueError}No model config found in the file
                # {OSError}SavedModel file does not exist
                # delete model when model file is not in hdf5 format but exists, maybe destroyed by interrupt
                if os.path.exists(self.model_hdf_file):
                    os.remove(self.model_hdf_file)
                self.init()
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        self.train(self.is_new)
        # self.plot()

    def init(self):
        self.model = tf.keras.Sequential()
        for i in range(len(self.core) - 1):
            self.model.add(tf.keras.layers.Dense(units=self.core[i + 1],
                                                 input_dim=self.core[i],
                                                 activation='relu'))
        self.model.add(tf.keras.layers.Dense(units=1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.mse)

    def train(self, is_new):
        start_time = time.time()
        if is_new:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_hdf_file,
                                                            monitor='loss',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            mode='min',
                                                            save_freq='epoch')
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                              patience=self.train_step // 100,
                                                              mode='min',
                                                              verbose=0,
                                                              restore_best_weights=True)
            # reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
            #                                               factor=0.5,
            #                                               patience=int(self.train_step // 100 * 0.9),
            #                                               verbose=0,
            #                                               mode='min',
            #                                               min_lr=self.learning_rate / 100)
            self.model.fit(self.train_x, self.train_y,
                           epochs=self.train_step,
                           initial_epoch=0,
                           batch_size=self.batch_size,
                           verbose=0,
                           callbacks=[checkpoint, early_stopping])
            min_err, max_err = self.get_err()
            err_length = max_err - min_err
            self.rename_model_file_by_err(err_length)
        else:
            min_err, max_err = self.get_err()
            err_length = max_err - min_err
        if not self.min_err or err_length < self.max_err - self.min_err:
            self.min_err = min_err
            self.max_err = max_err
            self.matrices = self.get_matrices()
        if self.use_threshold:
            if err_length > self.threshold:
                if self.retrain_times < self.retrain_time_limit:
                    self.init_model_name_by_random()
                    self.retrain_times += 1
                    self.logging.info("Retrain %d when score not perfect: Model %s, Err %f" % (
                        self.retrain_times, self.model_hdf_file, err_length))
                    self.train(True)
                else:
                    self.logging.info("Retrain time limit: Model %s, Err %f" % (self.model_hdf_file, err_length))
            else:
                self.logging.info("Model perfect: Model %s, Err %f" % (self.model_hdf_file, err_length))
        else:
            self.logging.info("Stop train when train early stop or epoch finish: Model %s, Err %f" % (
                self.model_hdf_file, err_length))
        end_time = time.time()
        self.logging.info("Model key: %s, Train time: %s" % (self.model_key, end_time - start_time))

    # 区别：simple结尾的函数用于模型的重训练
    def build_simple(self, matrices):
        self.init()
        if matrices:
            self.set_matrices(matrices)
        self.train_simple()

    def train_simple(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=self.train_step // 500,
                                                          mode='min',
                                                          verbose=0,
                                                          restore_best_weights=True)
        self.model.fit(self.train_x, self.train_y,
                       epochs=self.train_step,
                       initial_epoch=0,
                       batch_size=self.batch_size,
                       verbose=0,
                       callbacks=[early_stopping])
        self.matrices = self.get_matrices()
        self.min_err, self.max_err = self.get_err()

    def get_epochs(self):
        return len(self.model.history.epoch) if self.model.history else 0

    def get_matrices(self):
        return self.model.get_weights()

    def set_matrices(self, matrices):
        self.model.set_weights(matrices)

    # 对比mse/mae/mae+err_bound，最后选择mse+err_bound
    def mse(self, y_true, y_pred):
        diff = y_true - y_pred
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return mse_loss

    def mse_and_err_bound(self, y_true, y_pred):
        """
        自定义loss，用mse描述拟合程度，最大最小误差描述误差范围
        """
        # y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
        # diff_clip = y_true - y_pred_clip
        # range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
        diff = y_true - y_pred
        range_loss = tf.keras.backend.max(diff) - tf.keras.backend.min(diff)
        diff = y_true - y_pred
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return self.weight * range_loss + mse_loss

    def batch_predict(self):
        """
        分batch predict来减少内存占用
        避免一次性redict形成size(self.train_x) * 1的tensor造成内存溢出
        """
        train_x_len = len(self.train_x)
        step = 10000
        pres = np.empty(shape=(0, 1))
        for i in range(math.ceil(train_x_len / step)):
            tmp_pres = self.model(self.train_x[i * step:(i + 1) * step].reshape(-1, 1)).numpy()
            pres = np.vstack((pres, tmp_pres))
        return pres.flatten()

    def get_err(self):
        if self.train_y_max == self.train_y_min:
            return 0.0, 0.0
        pres = self.batch_predict()
        pres[pres < 0] = 0
        pres[pres > 1] = 1
        errs = (pres - self.train_y) * (self.train_y_max - self.train_y_min)
        return errs.min(), errs.max()

    def plot(self):
        pres = self.batch_predict()
        plt.plot(self.train_x, self.train_y, 'y--', label="true")
        plt.plot(self.train_x, pres, 'm--', label="predict")
        plt.legend()
        plt.savefig(self.model_hdf_file.replace("hdf5", "png").replace("hdf", "png"))
        plt.close()

    def get_best_model_file(self):
        """
        find the min err model path
        """
        min_err = 'best'
        files = os.listdir(self.model_hdf_dir)
        for file in files:
            tmp_split = file.split('.')
            tmp_model_key = tmp_split[0]
            if tmp_model_key == self.model_key:
                tmp_err = '.'.join(tmp_split[1:-1])
                try:
                    tmp_err = float(tmp_err)
                except:
                    continue
                if (min_err == 'best' or min_err > tmp_err) and tmp_err >= 0:
                    min_err = tmp_err
        best_file_name = '.'.join([self.model_key, str(min_err), "hdf5"])
        self.model_hdf_file = os.path.join(self.model_hdf_dir, best_file_name)

    def rename_model_file_by_err(self, err):
        """
        rename model file by err
        """
        new_file_name = '.'.join([self.model_key, str(err), "hdf5"])
        new_model_path = os.path.join(self.model_hdf_dir, new_file_name)
        try:
            os.rename(self.model_hdf_file, new_model_path)
        except FileExistsError:  # 相同误差的model已存在则删除旧模型
            os.remove(self.model_hdf_file)
        except FileNotFoundError:  # 原模型不存在则跳过
            pass
        self.model_hdf_file = new_model_path

    def init_model_name_by_random(self):
        """
        init model name by random float num
        """
        random_str = str(random.random() * -1)
        new_file_name = '.'.join([self.model_key, random_str, "hdf5"])
        self.model_hdf_file = os.path.join(self.model_hdf_dir, new_file_name)

    def clean_not_best_model_file(self):
        """
        delete all the model file besides the best one
        """
        min_err = 'best'
        files = os.listdir(self.model_hdf_dir)
        for file in files:
            tmp_split = file.split('.')
            tmp_model_key = tmp_split[0]
            if tmp_model_key == self.model_key:
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
                last_min_file_name = '.'.join([self.model_key, str(tmp_max_err), "hdf5"])
                if os.path.exists(os.path.join(self.model_hdf_dir, last_min_file_name)):
                    os.remove(os.path.join(self.model_hdf_dir, last_min_file_name))
                min_err = tmp_min_err
