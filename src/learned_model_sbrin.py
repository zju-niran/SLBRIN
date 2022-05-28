import logging
import os.path
import random
import time

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')  # 解决_tkinter.TclError: couldn't connect to display "localhost:11.0"


# Neural Network Model
class TrainedNN:
    def __init__(self, model_path, model_key, train_x, train_y, is_new, is_gpu, weight, core, train_step,
                 batch_size, learning_rate, use_threshold, threshold, retrain_time_limit):
        # common
        self.name = "Trained NN"
        self.model_path = model_path
        self.model_hdf_dir = os.path.join(model_path, "hdf/")
        self.model_key = model_key
        # data
        # 区别：train_x的是有序的，因此归一化不用计算最大最小值
        self.train_x_min = train_x[0]
        self.train_x_max = train_x[-1]
        self.train_x = (np.array(train_x) - self.train_x_min) / (self.train_x_max - self.train_x_min) - 0.5
        self.train_y_min = train_y[0]
        self.train_y_max = train_y[-1]
        self.train_y = (np.array(train_y) - self.train_y_min) / (self.train_y_max - self.train_y_min)
        # model structure
        self.is_new = is_new
        self.is_gpu = is_gpu
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

    def train(self):
        if self.is_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # 不输出报错：This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the
            # following CPU instructions in performance-critical operations:  AVX AVX2
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if self.is_new:
            self.init_model()
            self.model_hdf_file = os.path.join(self.model_hdf_dir, '.'.join([self.model_key, "best", "hdf5"]))
        else:
            self.get_best_model_file()
            try:
                self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
            except Exception:
                # {ValueError}No model config found in the file
                # {OSError}SavedModel file does not exist
                # delete model when model file is not in hdf5 format but exists, maybe destroyed by interrupt
                if os.path.exists(self.model_hdf_file):
                    os.remove(self.model_hdf_file)
                self.init_model()
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        self.train_model()

    def init_model(self):
        self.model = tf.keras.Sequential()
        for i in range(len(self.core) - 1):
            self.model.add(tf.keras.layers.Dense(units=self.core[i + 1],
                                                 input_dim=self.core[i],
                                                 activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(units=1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=self.score)

    def train_model(self):
        start_time = time.time()
        if self.is_new or self.use_threshold:
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
            self.model.fit(self.train_x, self.train_y,
                           epochs=self.train_step,
                           initial_epoch=0,
                           batch_size=self.batch_size,
                           verbose=0,
                           callbacks=[checkpoint, early_stopping])
            # 加载loss最小的模型
            self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
        min_err, max_err = self.get_err()
        err_length = max_err - min_err
        self.rename_model_file_by_err(err_length)
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
                    self.train_model()
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
    def train_simple(self, matrices):
        if self.is_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            # 不输出报错：This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the
            # following CPU instructions in performance-critical operations:  AVX AVX2
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.init_model()
        self.model_hdf_file = os.path.join(self.model_hdf_dir, self.model_key + ".best.hdf5")
        self.set_matrices(matrices)
        self.train_model_simple()

    def train_model_simple(self):
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
        self.model.fit(self.train_x, self.train_y,
                       epochs=self.train_step,
                       initial_epoch=0,
                       batch_size=self.batch_size,
                       verbose=0,
                       callbacks=[checkpoint, early_stopping])
        # 加载loss最小的模型
        self.model = tf.keras.models.load_model(self.model_hdf_file, custom_objects={'score': self.score})
        self.matrices = self.get_matrices()
        self.min_err, self.max_err = self.get_err()

    def get_matrices(self):
        return [np.mat(matrix) for matrix in self.model.get_weights()]

    def set_matrices(self, matrices):
        weights = [matrix.getA() for matrix in matrices]
        # bias的维度从(1, 128)/(1, 1)改为(128,)/(1,)
        for i in range(len(self.core)):
            weights[i * 2 + 1] = weights[i * 2 + 1].flatten()
        self.model.set_weights(weights)

    def score(self, y_true, y_pred):
        """
        自定义loss，用mse描述拟合程度，最大最小误差描述误差范围
        TODO: checkpoint保存下来的最优模型(loss最小)不一定是误差范围小的
        """
        # 对比mse/mae/mae+minmax，最后选择mse+minmax
        # 这里的y应该是局部的，因此scores和err算出来不一致
        y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
        diff_clip = y_true - y_pred_clip
        range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
        diff = y_true - y_pred
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return self.weight * range_loss + mse_loss

    def get_err(self):
        # 区别：计算err的时候不考虑breakpoints
        inputs = self.train_x[1:-1]
        input_len = len(inputs)
        pres = self.model(inputs).numpy().flatten()
        pres[pres < 0] = 0
        pres[pres > 1] = 1
        errs = pres * (input_len - 1) - np.arange(input_len)
        return errs.min(), errs.max()

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
