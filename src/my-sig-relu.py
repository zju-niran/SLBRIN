import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 开启GPU
from src.spatial_index.common_utils import Region, ZOrder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 读取数据
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# load data
path = '../data/trip_data_1_100000.csv'
data = pd.read_csv(path)
z_order = ZOrder(data_precision=6, region=Region(40, 42, -75, -73))
z_values = data.apply(lambda t: z_order.point_to_z(t.x, t.y), 1)
x_data = z_values.sort_values(ascending=True).values
y_data = pd.Series(np.arange(0, len(data)) / 100).values
divisor = 100 * 1.0 / (len(data) / 100)
y_data = (y_data * divisor).astype(int)
# 输入输出归一化
# x_data = data[1].values
x_data_min = x_data.min()
x_data_max = x_data.max()
# y_data = data[2].values
y_data_min = y_data.min()
y_data_max = y_data.max()
x_data = (x_data - x_data_min) / (x_data_max - x_data_min)
y_data = (y_data - y_data_min) / (y_data_max - y_data_min)
x_data = np.array([462729363013603.0]).astype("float")
y_data = np.array([99951])
# 构建模型
model = Sequential()
# 1-10-1，添加一个隐藏层
model.add(Dense(units=128, input_dim=1, activation='sigmoid'))  # units是隐藏层，输出维度，输出y，input_dim是输入维度，输入x
model.add(Dense(units=1, input_dim=128, activation='relu'))  # input_dim可以不写，它可以识别到上一句的输出是10维


# 定义优化器
def score(y_true, y_pred):
    # 这里的y应该是局部的，因此scores和err算出来不一致
    # clip的时候也得用float，不然y_true - y_pred_clip的时候报错
    # TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int32 of argument 'x'.
    y_pred_clip = tf.keras.backend.clip(y_pred, 0, 1)
    diff_clip = y_true - y_pred_clip
    range_loss = tf.keras.backend.max(diff_clip) - tf.keras.backend.min(diff_clip)
    diff = y_true - y_pred
    mse_loss = tf.keras.backend.mean(tf.keras.backend.square(diff), axis=-1)
    return 0.1 * range_loss + mse_loss


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=score)  # 编译这个模型，sgd是随机梯度下降法，优化器.mse是均方误差
# 定义早停
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=500,
                                                  mode='min',
                                                  verbose=2)
# 定义tensorboard
tbcb = tf.keras.callbacks.TensorBoard(log_dir="./tb_model_save_dir", histogram_freq=1, write_grads=True)
callbacks_list = [early_stopping, tbcb]
# 训练
history = model.fit(x_data, y_data,
                    epochs=2000,
                    initial_epoch=0,
                    batch_size=1024,
                    verbose=2,
                    callbacks=callbacks_list)

# 打印pred和y
y_pred = model.predict(x_data)
plt.scatter(x_data, y_data, label="y_data")
plt.plot(x_data, y_pred, 'r-', lw=3, label="y_pred")  # r-表示红色的线，lw表示线宽
plt.legend()
plt.show()
# 计算极值
y_pred = model.predict(x_data).flatten()
diff = y_data - y_pred
diff_length = diff.max() - diff.min()
print(diff_length)
