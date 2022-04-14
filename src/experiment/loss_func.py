import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

sys.path.append('/home/zju/wlj/st-learned-index')


def my_score(y_true, y_pred):
    y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
    diff_clip = y_true - y_pred_clip
    range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
    diff = y_true - y_pred
    mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
    return 1 * range_loss + mse_loss


def train(batch_size, lr, loss):
    model = Sequential()
    model.add(Dense(units=128, input_dim=1, activation='sigmoid'))
    model.add(Dense(units=1))

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=loss)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=50,
                                                      mode='min',
                                                      verbose=0)
    # tbcb = tf.keras.callbacks.TensorBoard(log_dir="./tb_model_save_dir", histogram_freq=1, write_grads=True)
    callbacks_list = [early_stopping]
    history = model.fit(x_data, y_data,
                        epochs=5000,
                        initial_epoch=0,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=callbacks_list)
    return model, history.epoch[-1]


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # 1. 读取数据
    x_data = np.load('1.npy', allow_pickle=True)
    y_data = np.load('2.npy', allow_pickle=True)
    # 输入输出归一化
    x_data_min = x_data.min()
    x_data_max = x_data.max()
    y_data_min = y_data.min()
    y_data_max = y_data.max()
    x_data = (x_data - x_data_min) / (x_data_max - x_data_min) - 0.5
    y_data = (y_data - y_data_min) / (y_data_max - y_data_min)
    # 2. 设置实验参数
    loss_funcs = [my_score, tf.keras.losses.mse, tf.keras.losses.mae, tf.keras.losses.log_cosh]
    batch_sizes = [512, 256, 128, 64, 32, 16, 8, 4, 2]
    # 3. 开始实验
    for loss_func in loss_funcs:
        for bs in batch_sizes:
            start_time = time.time()
            model, epoch = train(bs, 0.1, loss_func)
            end_time = time.time()
            y_pred = model.predict(x_data).flatten()
            diff = y_data - y_pred
            diff_length = diff.max() - diff.min()
            print("loss func: %s, batch size: %s, diff: %s, epoch: %s, time: %s"
                  % (loss_func.__name__, bs, diff_length, epoch, end_time - start_time))
