#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 手写字体 逻辑回归(权重+偏置)  
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data   # 读取mnist手写字体数据工具

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

## 模型函数 ####
def model(X, w, b):
    return tf.matmul(X, w) + b 
# notice we use the same model as linear regression, 
# this is because there is a baked in cost function which performs softmax and cross entropy

## 数据处理 data preprocessing
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

## 训练数据的 占位符
X = tf.placeholder("float", [None, 784]) # a picture is a 784 dimension verb
Y = tf.placeholder("float", [None, 10])  # a label is a 10dimension verb which has only one 1 ,others is 0

## 权重初始化
w = init_weights([784, 10]) # 784*10 输入n×784  .* 784*10 -> n*10 最后每张图像输出 1*10
## 偏置 
b = tf.Variable(tf.zeros([10])) # 1*10
## 模型
pred_y_x = model(X, w, b)

#(logits, labels, name=None)   cross_entropy = -tf.reduce_sum(y_real*tf.log(y_predict))
## 计算代价函数 softmax回归 后 对数 信息增益 均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_y_x, Y)) 
## 优化函数        梯度下降法 学习率为0.05
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
## 预测值
predict_op = tf.argmax(pred_y_x, 1) # 逻辑回归输出（1*10）中的最大值即为 预测数字 

## 变量初始化
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

## 启动 图 回话
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    #tf.global_variables_initializer().run() 
    sess.run(init)

    for i in range(50):#训练50次
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))#计算每次训练的正确率



