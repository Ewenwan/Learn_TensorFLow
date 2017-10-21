#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 三维变量 线性回归 linear regression
import tensorflow as tf
import numpy as np

# 生成数据 generate date
x_data = np.float32(np.random.rand(2, 100))    # 2行 100列
## 预测 两个权重 0.1 和 0.2 以及一个 偏置 0.3
# 无噪声
y_data = np.dot([0.100, 0.200], x_data) + 0.300# y=a*x1 + b*x2 +c
# 有噪声
#y_data = np.dot([0.100, 0.200], x_data) + 0.300 + np.random.randn(*(1,x_data.shape[1]))*0.11

# 创建模型 creat model
b = tf.Variable(tf.zeros([1]))#偏置变量 c 初始化为0
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))#1行2列 权重变量 初始化为随机 -1~1之间均匀分布
y_model = tf.matmul(W, x_data) + b#模型

# 代价函数 coss function
loss = tf.reduce_mean(tf.square(y_model - y_data))#差值平方 均值

# 优化函数  梯度下降法 学习率为0.55
train_op = tf.train.GradientDescentOptimizer(0.55).minimize(loss)
#train = optimizer.minimize(loss)

# # you need to initialize variables (in this case just variable W)
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

# Launch the graph in a sessi
with tf.Session() as sess:
    sess.run(init)#初始化变量
# 开始训练loop train
    for step in xrange(0, 201):#训练200次 
        sess.run(train_op)#执行优化函数
        if step % 20 == 0:#没20步 显示一下结果
            print step, sess.run(W), sess.run(b)

# you will get the result W: [[0.100  0.200]], b: [0.300]
