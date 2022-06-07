import math
import os

import numpy as np
import tensorflow as tf

from src.spatial_index.common_utils import normalize_output, normalize_input


class TrainedNN_Simple:
    def __init__(self, train_x, train_y, is_gpu, weight, core, train_step, batch_size, learning_rate):
        # common
        self.name = "Trained NN"
        # data
        self.train_x, self.train_x_min, self.train_x_max = normalize_input(np.array(train_x).astype("float"))
        self.train_y, self.train_y_min, self.train_y_max = normalize_output(np.array(train_y).astype("float"))
        # model structure
        self.is_gpu = is_gpu
        self.weight = weight
        self.core = core
        self.train_step = train_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # for compute
        self.model = None
        self.matrices = None
        self.min_err = None
        self.max_err = None

    def train(self):
        if self.is_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.init_model()
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
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=50,
                                                          mode='min',
                                                          verbose=0)
        self.model.fit(self.train_x, self.train_y,
                       epochs=self.train_step,
                       initial_epoch=0,
                       batch_size=self.batch_size,
                       verbose=0,
                       callbacks=[early_stopping])
        self.min_err, self.max_err = self.get_err()
        self.matrices = self.get_matrices()

    def get_matrices(self):
        return self.model.get_weights()

    def set_matrices(self, matrices):
        self.model.set_weights(matrices)

    def score(self, y_true, y_pred):
        y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
        diff_clip = y_true - y_pred_clip
        range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
        diff = y_true - y_pred
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return self.weight * range_loss + mse_loss

    def batch_predict(self):
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
