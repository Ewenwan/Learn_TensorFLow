#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 简单的网络
# 多层感知机（MLP，Multilayer Perceptron）也叫
# 人工神经网络（ANN，Artificial Neural Network），
# 除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构
# http://blog.csdn.net/xueli1991/article/details/52386611
# 各层有权重也有偏置
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

## 初始化偏置 #####
def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

## 模型函数 带 偏置 ####
def model(X, w_h, w_o, b_h, b_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h) # 隐含层 sigmoid激活函数
    return tf.matmul(h, w_o) + b_o             # 输出层 未包含 softmax损失函数 cost fn 做了

## 数据处理 data preprocessing
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

## 训练数据的 占位符
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

### 模型参数初始化 ####
w_h = init_weights([784, 625]) # 隐含层权重 hidden layer weight  n×784  .* 784*625 -> n*625
w_o = init_weights([625, 10])  # 输出层权重 output layer weight  n*625  .* 625*10  -> n*10  最后每张图像输出 1*10
b_h = init_bias([625]);        # 隐含层偏置 hidden layer bias    n*625 + 1*625
b_o = init_bias([10]);         # 输出层偏置 output layer bias    n*10  + 1*10  

## 模型
predict_y_x = model(X, w_h, w_o, b_h, b_o)

## 计算代价函数 softmax回归 后 对数 信息增益 均值
#(logits, labels, name=None)   cross_entropy = -tf.reduce_sum(y_real*tf.log(y_predict))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict_y_x, Y)) # compute costs
# 优化函数        梯度下降法 学习率为0.05
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
## 预测值
predict_op = tf.argmax(predict_y_x, 1)

## 变量初始化
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

## 启动 图 回话  Launch the graph in a session
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    #tf.global_variables_initializer().run() 
    sess.run(init)
    for i in range(20):#训练20次
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
		# start=(0 128 128+128 ...) end =(128 128+128 128+128+128 ...)
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))#计算每次训练的正确率





