import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 不输出报错：This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the
# following CPU instructions in performance-critical operations:  AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

cont = 100000
learning_rate = 0.01
epochs = 5000
batch_size = 1024
early_stopping_patience = epochs / 500
input_x = np.linspace(-1, 1, cont)  # 创建等差数列
# input_y = 3*np.sin(input_x) + np.random.uniform(0,1,cont)
input_y = input_x * 1.0 + \
          pow(input_x, 2) * 2.0 + \
          pow(input_x, 3) * 1.5 + \
          pow(input_x, 4) * 1.1 + \
          5.0 + \
          np.random.uniform(-0.5, 0.5, cont)

# 绘制出原始数据
plt.scatter(input_x, input_y)
plt.show()

initializer = tf.initializers.GlorotUniform()
# W = tf.Variable(initializer(shape=(1, ), dtype=tf.float64), name="W")
# b = tf.Variable(initializer(shape=(1, ), dtype=tf.float64), name="b")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=128,
                                input_dim=1,
                                activation='relu'))
model.add(tf.keras.layers.Dense(units=1))
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="mse")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                  patience=early_stopping_patience,
                                                  mode='min',
                                                  verbose=0,
                                                 restore_best_weights=True)
plot_epoches = epochs // 100
for i in range(10):
    model.fit(input_x, input_y,
              epochs=plot_epoches,
              initial_epoch=0,
              batch_size=batch_size,
              verbose=1,
              callbacks=[early_stopping])
    pred_y = model(input_x)
    # 作图，显示线性回归的结果
    plt.plot(input_x, input_y, 'bo', label='real data')
    plt.plot(input_x, pred_y, 'r', label='predicted data')
    plt.legend()
    plt.show()
