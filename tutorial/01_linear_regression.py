#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 两维变量 线性回归 linear regression
import tensorflow as tf#tf包
import numpy as np     #

trX = np.linspace(-1, 1, 201)   # -1 到 200的线性序列 步长为1
trY = 4 * trX + 2 + np.random.randn(*trX.shape) * 0.33 
# 创建一个y =4*x+4 带有随机数

X = tf.placeholder("float") # float类占位符 X
Y = tf.placeholder("float") # float类占位符 Y

##模型函数
#简单的线性回归模型 linear regression model
def model(X, w, b):          
    return tf.mul(X, w) - b 

##需要估计的变量
b = tf.Variable(0.0, name="bias")    # 变量 偏置
w = tf.Variable(0.0, name="weights") # 变量 权重
# 模型
y_model = model(X, w, b)
#代价函数CF
cost = tf.square(Y - y_model)        #误差平方为 代价函数 cost function

# 优化函数  梯度下降法 学习率为0.007
train_op = tf.train.GradientDescentOptimizer(0.007).minimize(cost) 

# 开始回话 执行优化
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    tf.global_variables_initializer().run() 

    for i in range(200):
        for (x, y) in zip(trX, trY):#对 
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))   # 优化后的 权重值 接近 4
    print(sess.run(b))   # 优化后的 偏置值 接近  -2
