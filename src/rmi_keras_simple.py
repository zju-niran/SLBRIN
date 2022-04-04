import logging
import os.path
import time

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use('Agg')  # 解决_tkinter.TclError: couldn't connect to display "localhost:11.0"

from src.spatial_index.common_utils import nparray_normalize, nparray_diff_normalize_reverse_arr, \
    nparray_normalize_reverse_arr


class TrainedNN:
    def __init__(self, model_path, model_index, train_x, train_y, cores, train_step_num, batch_size, learning_rate):
        self.name = "Trained NN Simple"
        if cores is None:
            cores = []
        self.core_nums = cores
        self.train_step_nums = train_step_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_x, self.train_x_min, self.train_x_max = nparray_normalize(np.array(train_x).astype("float"))
        self.train_y, self.train_y_min, self.train_y_max = nparray_normalize(np.array(train_y).astype("float"))
        self.model = None
        self.model_path = model_path
        self.model_index = model_index
        self.min_err, self.max_err = 0, 0
        self.weights = None
        logging.basicConfig(filename=os.path.join(self.model_path, "log.file"),
                            level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y/%m/%d %H:%M:%S %p")
        self.logging = logging.getLogger(self.name)

    # train model
    def train(self):
        start_time = time.time()
        # GPU配置
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # 不输出报错：This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the
        # following CPU instructions in performance-critical operations:  AVX AVX2
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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
                                                          patience=500,
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
        self.weights = self.get_weights()
        end_time = time.time()
        self.logging.info("Model index: %s, Train time: %s" % (self.model_index, end_time - start_time))

    def get_weights(self):
        return [np.mat(weight) for weight in self.model.get_weights()]

    def score(self, y_true, y_pred):
        y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
        diff_clip = y_true - y_pred_clip
        range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
        diff = y_true - y_pred
        mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
        return 0.1 * range_loss + mse_loss

    def get_err(self):
        pres = self.model(self.train_x).numpy().flatten()
        return nparray_diff_normalize_reverse_arr(pres, self.train_y, self.train_y_min, self.train_y_max)

    def predict(self):
        pres = self.model(self.train_x).numpy().flatten()
        return nparray_normalize_reverse_arr(pres, self.train_y_min, self.train_y_max)
