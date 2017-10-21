#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 多层网络 两层 隐含层
# 权重 + 偏置
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
## 初始化偏置 #####
def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    
## 模型函数 权重+偏置 ##   
def model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p_keep_input, p_keep_hidden): 
    # this network is the same as the previous one except with an extra hidden layer + dropout
    ## 输入层 ##
    X = tf.nn.dropout(X, p_keep_input)   # 输入层 dropout  激活部分神经元
    ## 第一层
    h = tf.nn.relu(tf.matmul(X, w_h) + b_h)# 非线性激活函数 nonlinear activate function  relu / sigmoid  / maxout
    h = tf.nn.dropout(h, p_keep_hidden)  # 第一层 dropout  激活部分神经元
    ## 第二层 ##
    h2 = tf.nn.relu(tf.matmul(h, w_h2) + b_h2)# 非线性激活函数 nonlinear activate function  relu / sigmoid  / maxout
    h2 = tf.nn.dropout(h2, p_keep_hidden) # 第二层dropout  激活部分神经元

    return tf.matmul(h2, w_o) + b_o#输出层


## 数据处理 data preprocessing
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

## 训练数据的 占位符
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

### 模型参数初始化 ####
w_h = init_weights([784, 625]) # 第一层权重
w_h2 = init_weights([625, 625])# 第二层权重
w_o = init_weights([625, 10])  # 输出层权重
b_h = init_bias([625]);        # 第一层偏置 hidden layer bias
b_h2 = init_bias([625]);       # 第二层偏置 hidden layer bias
b_o = init_bias([10]);         # 输出层偏置 

## dropout参数配置  占位符 优化时再 调入
p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

## 模型 
pred_y_x = model(X, w_h, b_h, w_h2, b_h2, w_o, b_o, p_keep_input, p_keep_hidden)

## 计算代价函数 softmax回归 后 对数 信息增益 均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_y_x, Y))
# 优化函数   全局学习速率 0.001 
# http://blog.csdn.net/u014595019/article/details/52989301
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)   # ReLU can not use Gradient-Based method 
## 预测值
predict_op = tf.argmax(pred_y_x, 1)

## 变量初始化
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

## 启动 图 回话  Launch the graph in a session
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    #tf.global_variables_initializer().run() 
    sess.run(init)

    for i in range(30):#训练30次
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
		# start=(0 128 128+128 ...) end =(128 128+128 128+128+128 ...)

            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})# 指定 dropout
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_input: 1.0,   ## 测试时关闭 dropout
                                                         p_keep_hidden: 1.0})))#计算每次训练的正确率


