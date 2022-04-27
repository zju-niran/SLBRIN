import logging
import math
import os.path
import random
import shutil
import time

import matplotlib
import numpy as np
import tensorflow as tf

from src.spatial_index.common_utils import normalize_output, normalize_input, denormalize_diff_minmax

matplotlib.use('Agg')  # 解决_tkinter.TclError: couldn't connect to display "localhost:11.0"
import matplotlib.pyplot as plt


# Neural Network Model
class TrainedNN:
    def __init__(self, model_path, model_key, train_x, train_y, use_threshold, threshold, cores, train_step_num,
                 batch_size, learning_rate, retrain_time_limit, weight):
        self.name = "Trained NN"
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # 当只有一个输入输出时，整数的key作为y_true会导致score中y_true-y_pred出现类型错误：
        # TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
        self.train_x, self.train_x_min, self.train_x_max = normalize_input(np.array(train_x).astype("float"))
        self.train_y, self.train_y_min, self.train_y_max = normalize_output(np.array(train_y).astype("float"))
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.model = None
        self.min_err, self.max_err = 0, 0
        self.weights = None
        self.retrain_times = 0
        self.retrain_time_limit = retrain_time_limit
        self.model_path = model_path
        self.model_key = model_key
        self.model_hdf_dir = os.path.join(model_path, "hdf/")
        self.model_png_dir = os.path.join(model_path, "png/")
        self.model_loss_dir = os.path.join(model_path, "loss/")
        if os.path.exists(self.model_hdf_dir) is False:
            os.makedirs(self.model_hdf_dir)
        if os.path.exists(self.model_png_dir) is False:
            os.makedirs(self.model_png_dir)
        if os.path.exists(self.model_loss_dir) is False:
            os.makedirs(self.model_loss_dir)
        self.model_hdf_file = os.path.join(self.model_hdf_dir, self.model_key + "_weights.best.hdf5")
        self.model_png_file = os.path.join(self.model_png_dir, self.model_key + "_weights.best.png")
        self.model_loss_file = os.path.join(self.model_loss_dir, self.model_key + "_weights.best.loss")
        self.clean_not_best_model_file()
        self.get_best_model_file()
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        self.weight = weight

    # train model
    def train(self):
        start_time = time.time()
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
            model.add(tf.keras.layers.Dense(units=self.core_nums[-1]))
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
                                                              patience=50,
                                                              mode='min',
                                                              verbose=0)
            # reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
            #                                               factor=0.5,
            #                                               patience=45,
            #                                               verbose=0,
            #                                               mode='min',
            #                                               min_lr=0.0001)
            # tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.model_loss_file,
            #                                              histogram_freq=1)
            callbacks_list = [checkpoint, early_stopping]
            # fit and save model
            history = self.model.fit(self.train_x, self.train_y,
                                     epochs=self.train_step_nums,
                                     initial_epoch=0,
                                     batch_size=self.batch_size,
                                     verbose=0,
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
                    self.logging.info("Retrain %d when score not perfect: Model %s, Err %f, Threshold %f" % (
                        self.retrain_times, self.model_hdf_file, err_length, self.threshold))
                    self.train()
                else:
                    self.logging.info("Retrain time limit: Model %s, Err %f, Threshold %f" % (
                        self.model_hdf_file, err_length, self.threshold))
                    self.get_best_model_file()
                    self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
                    self.min_err, self.max_err = self.get_err()
            else:
                self.logging.info("Model perfect: Model %s, Err %f, Threshold %f" % (
                    self.model_hdf_file, err_length, self.threshold))
        else:
            self.logging.info("Stop train when train early stop or epoch finish: Model %s, Err %f, Threshold %f" % (
                self.model_hdf_file, err_length, self.threshold))
        self.weights = self.get_weights()
        end_time = time.time()
        self.logging.info("Model key: %s, Train time: %s" % (self.model_key, end_time - start_time))

    def is_model_file_valid(self):
        try:
            model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
            return True
        except Exception:
            # {ValueError}No model config found in the file
            # {OSError}SavedModel file does not exist
            return False

    def get_weights(self):
        return [np.mat(weight) for weight in self.model.get_weights()]

    def score(self, y_true, y_pred):
        # 对比mse/mae/mae+minmax，最后选择mse+minmax
        # 这里的y应该是局部的，因此scores和err算出来不一致
        y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
        diff_clip = y_true - y_pred_clip
        range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
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
        return denormalize_diff_minmax(self.batch_predict(), self.train_y, self.train_y_min, self.train_y_max)

    def plot(self):
        pres = self.batch_predict()
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
        best_png_name = '.'.join([self.model_key, str(min_err), "png"])
        best_loss_name = '.'.join([self.model_key, str(min_err), "loss"])
        self.model_hdf_file = os.path.join(self.model_hdf_dir, best_file_name)
        self.model_png_file = os.path.join(self.model_png_dir, best_png_name)
        self.model_loss_file = os.path.join(self.model_loss_dir, best_loss_name)

    def rename_model_file_by_err(self, err):
        """
        rename model file by err
        :param err: float
        """
        new_file_name = '.'.join([self.model_key, str(err), "hdf5"])
        new_png_name = '.'.join([self.model_key, str(err), "png"])
        new_loss_name = '.'.join([self.model_key, str(err), "loss"])
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
        new_file_name = '.'.join([self.model_key, random_str, "hdf5"])
        new_png_name = '.'.join([self.model_key, random_str, "png"])
        new_loss_name = '.'.join([self.model_key, random_str, "loss"])
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
                last_min_png_name = '.'.join([self.model_key, str(tmp_max_err), "png"])
                last_min_loss_name = '.'.join([self.model_key, str(tmp_max_err), "loss"])
                if os.path.exists(os.path.join(self.model_hdf_dir, last_min_file_name)):
                    os.remove(os.path.join(self.model_hdf_dir, last_min_file_name))
                if os.path.exists(os.path.join(self.model_png_dir, last_min_png_name)):
                    os.remove(os.path.join(self.model_png_dir, last_min_png_name))
                if os.path.exists(os.path.join(self.model_loss_dir, last_min_loss_name)):
                    shutil.rmtree(os.path.join(self.model_loss_dir, last_min_loss_name))
                min_err = tmp_min_err
