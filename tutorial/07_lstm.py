#-*- coding:utf-8 -*-
#!/usr/bin/env python
#Inspired by https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
#LSTM 详解 http://blog.csdn.net/u014595019/article/details/52605693
# http://baijiahao.baidu.com/s?id=1579839249811513976&wfr=spider&for=pc
import tensorflow as tf
import numpy as np
#import input_data

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28
'''
inputs 引数接受形态为 [batch_size,input_size] 的张量列表。
列表的长度为将网络展开后的时间步数，即列表中每一个元素都分别对应网络展开的时间步。
比如在 MNIST 数据集中，我们有 28x28 像素的图像，
每一张都可以看成拥有 28 行 28 个像素的图像。
我们将网络按 28 个时间步展开，以使在每一个时间步中，可以输入一行 28 个像素（input_size），
从而经过 28 个时间步输入整张图像。给定图像的 batch_size 值，则每一个时间步将分别收到 batch_size 个图像。
'''
## 变量配置
# configuration variables
input_vec_size = lstm_size = 28
# lstm_size 一个 LSTM 单元格中的单元数（标准LSTM单元）
time_step_size = 28# lstm 记录的时间步长

batch_size = 128#每一次训练 batch大小 
test_size = 256 #每一次测试 batch大小

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

## 模型 ##
def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(0, time_step_size, XR) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
    # TensorFlow 中，基础的 LSTM 单元格
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

## 数据处理 
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# reshape成 lstm网络需要的 图片格式
trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)
## 训练数据的 占位符
X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

# get lstm_size and output 10 labels
## 初始化权重和偏置
W = init_weights([lstm_size, 10])
B = init_weights([10])

## 模型
py_x, state_size = model(X, W, B, lstm_size)

## 计算代价函数 softmax回归 后 对数 信息增益 均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))

## 训练优化操作
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
## 预测值
predict_op = tf.argmax(py_x, 1)

## 启动 图 回话  Launch the graph in a session
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(100):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
		# start=(0 128 128+128 ...) end =(128 128+128 128+128+128 ...)
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]#测试数据

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices]})))#计算每次训练的正确率



