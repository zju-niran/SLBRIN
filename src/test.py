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

w1 = tf.Variable(0.0, dtype=tf.float64, name='w1')
w2 = tf.Variable(0.0, dtype=tf.float64, name='w2')
w3 = tf.Variable(0.0, dtype=tf.float64, name='w3')
w4 = tf.Variable(0.0, dtype=tf.float64, name='w4')
b = tf.Variable(0.0, dtype=tf.float64, name='b')
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
best_model = None
min_loss = tf.Variable(float("inf"), dtype=tf.float64, name='el')
early_stopping_patience_epochs = 0
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, cont, batch_size):
        start = i
        end = i + batch_size
        x = input_x[start:end]
        y = input_y[start:end]
        with tf.GradientTape() as tape:
            y_predict = tf.multiply(w1, x) + \
                        tf.multiply(w2, tf.pow(x, 2)) + \
                        tf.multiply(w3, tf.pow(x, 3)) + \
                        tf.multiply(w4, tf.pow(x, 4)) + \
                        b
            batch_loss = tf.reduce_mean(tf.math.pow(y_predict - y, 2))
            epoch_loss += batch_loss.numpy()
        train_variables = [w1, w2, w3, w4, b]
        gradients = tape.gradient(batch_loss, train_variables)
        optimizer.apply_gradients(zip(gradients, train_variables))
    if epoch_loss < min_loss:
        min_loss = epoch_loss
        best_model = [w1.numpy(), w2.numpy(), w3.numpy(), w4.numpy(), b.numpy()]
        early_stopping_patience_epochs = 0
    else:
        early_stopping_patience_epochs += 1
        if early_stopping_patience_epochs >= early_stopping_patience:
            print("early stop")
            break
    print(epoch, epoch_loss)

print(w1)
print(w2)
print(w3)
print(w4)
print(b)

# 作图，显示线性回归的结果
plt.plot(input_x, input_y, 'bo', label='real data')
plt.plot(input_x, input_x * w1 + input_x ** 2 * w2 + input_x ** 3 * w3 + input_x ** 4 * w4 + b, 'r',
         label='predicted data')
plt.legend()
plt.show()
