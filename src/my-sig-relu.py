import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 开启GPU

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 读取数据
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# load data
x_data = np.load('1.npy', allow_pickle=True)
y_data = np.load('2.npy', allow_pickle=True)
# 输入输出归一化
# x_data = data[1].values
x_data_min = x_data.min()
x_data_max = x_data.max()
# y_data = data[2].values
y_data_min = y_data.min()
y_data_max = y_data.max()
x_data = (x_data - x_data_min) / (x_data_max - x_data_min)
y_data = (y_data - y_data_min) / (y_data_max - y_data_min)


# 定义优化器
def my_score(y_true, y_pred):
    y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
    diff_clip = y_true - y_pred_clip
    range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
    diff = y_true - y_pred
    mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
    return 0.1 * range_loss + mse_loss


def mae_score(y_true, y_pred):
    diff = y_true - y_pred
    mse_loss = tf.keras.backend.mean(tf.keras.backend.abs(diff), axis=-1)
    return mse_loss


def mse_score(y_true, y_pred):
    diff = y_true - y_pred
    mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
    return mse_loss


def ce_score(y_true, y_pred):
    return tf.keras.backend.sum(y_true * tf.keras.backend.log(y_pred) +
                                (1 - y_true) * tf.keras.backend.log(1 - y_pred))


def train(batch_size, lr):
    # 构建模型
    model = Sequential()
    # 1-10-1，添加一个隐藏层
    model.add(Dense(units=128, input_dim=1, activation='sigmoid'))
    model.add(Dense(units=1))

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                  loss=my_score)
    # 定义早停
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                      patience=50,
                                                      mode='min',
                                                      verbose=0)
    # 定义tensorboard
    # tbcb = tf.keras.callbacks.TensorBoard(log_dir="./tb_model_save_dir", histogram_freq=1, write_grads=True)
    callbacks_list = [early_stopping]
    # 训练
    history = model.fit(x_data, y_data,
                        epochs=5000,
                        initial_epoch=0,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=callbacks_list)
    return model, history.epoch[-1]


lrs = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for lr in lrs:
    for batch_size in batch_sizes:
        model, epoch = train(batch_size, lr)
        # 打印pred和y
        # y_pred = model.predict(x_data)
        # plt.scatter(x_data, y_data, label="y_data")
        # plt.plot(x_data, y_pred, 'r-', lw=3, label="y_pred")  # r-表示红色的线，lw表示线宽
        # plt.legend()
        # plt.show()
        # 计算极值
        y_pred = model.predict(x_data).flatten()
        diff = y_data - y_pred
        diff_length = diff.max() - diff.min()
        print("lr: %s, batch_size: %s, diff: %s, epoch: %s" %
              (lr, batch_size, diff_length, epoch))
