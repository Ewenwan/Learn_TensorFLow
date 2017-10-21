#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 简单的网络
# 多层感知机（MLP，Multilayer Perceptron）也叫
# 人工神经网络（ANN，Artificial Neural Network），
# 除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构
# http://blog.csdn.net/xueli1991/article/details/52386611
# 各层仅有权重无偏置
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

## 模型函数 无 偏置 ####
def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # 隐含层 sigmoid激活函数
    return tf.matmul(h, w_o)             # 输出层 未包含 softmax损失函数 cost fn 做了

## 数据处理 data preprocessing
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

## 训练数据的 占位符
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

## 权重初始化
w_h = init_weights([784, 625]) # 隐含层 权重  n×784  .* 784*625 -> n*625
w_o = init_weights([625, 10])  # 输出层 权重  n*625  .* 625*10  -> n*10  最后每张图像输出 1*10
## 模型
py_x = model(X, w_h, w_o)

## 计算代价函数 softmax回归 后 对数 信息增益 均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
# 优化函数        梯度下降法 学习率为0.05
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
## 预测值
predict_op = tf.argmax(py_x, 1) # 逻辑回归输出（1*10）中的最大值即为 预测数字 

## 变量初始化
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

## 启动 图 回话
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    #tf.global_variables_initializer().run() 
    sess.run(init)

    for i in range(30):#训练30次
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
             # start=(0 128 128+128 ...) end =(128 128+128 128+128+128 ...)
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))#计算每次训练的正确率

