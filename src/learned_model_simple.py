import logging
import math
import os.path
import time

import numpy as np
import tensorflow as tf

from src.spatial_index.common_utils import normalize_input, normalize_output, denormalize_diff_minmax


class TrainedNN:
    def __init__(self, model_path, model_key, train_x, train_y, cores, train_step_num, batch_size, learning_rate,
                 weight):
        self.name = "Trained NN Simple"
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_x, self.train_x_min, self.train_x_max = normalize_input(np.array(train_x).astype("float"))
        self.train_y, self.train_y_min, self.train_y_max = normalize_output(np.array(train_y).astype("float"))
        self.model = None
        self.model_path = model_path
        self.model_key = model_key
        self.min_err, self.max_err = 0, 0
        self.matrices = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)
        self.weight = weight

    # train model
    def train(self, is_gpu):
        start_time = time.time()
        if is_gpu:
            # GPU配置
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
        model = tf.keras.Sequential()
        for i in range(len(self.core_nums) - 2):
            model.add(tf.keras.layers.Dense(units=self.core_nums[i + 1],
                                            input_dim=self.core_nums[i],
                                            activation='sigmoid'))
        model.add(tf.keras.layers.Dense(units=self.core_nums[-1]))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer, loss=self.score)
        self.model = model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=50,
                                                          mode='min',
                                                          verbose=0)
        callbacks_list = [early_stopping]
        self.model.fit(self.train_x, self.train_y,
                       epochs=self.train_step_nums,
                       initial_epoch=0,
                       batch_size=self.batch_size,
                       verbose=0,
                       callbacks=callbacks_list)
        self.min_err, self.max_err = self.get_err()
        self.matrices = self.get_matrices()
        end_time = time.time()
        self.logging.info("Model key: %s, Train time: %s" % (self.model_key, end_time - start_time))

    def get_matrices(self):
        return [np.mat(matrix) for matrix in self.model.get_weights()]

    def score(self, y_true, y_pred):
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
