#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 三维变量 线性回归  + tensorboard 显示优化记录
import tensorflow as tf
import numpy as np

# 生成数据 generate date
x_data = np.float32(np.random.rand(2, 100))    # 2行 100列 随机数
## 预测 两个权重 0.1 和 0.2 以及一个 偏置 0.3
# 无噪声
y_data = np.dot([0.100, 0.200], x_data) + 0.300# y=a*x1 + b*x2 +c
# 有噪声
#y_data = np.dot([0.100, 0.200], x_data) + 0.300 + np.random.randn(*(1,x_data.shape[1]))*0.11

# 创建模型 creat model
b = tf.Variable(tf.zeros([1]), name="b")#偏置变量 c 初始化为0
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name="W")#1行2列 权重变量 初始化为随机 -1~1之间均匀分布
#y_model = tf.matmul(W, x_data) + b#模型
y_model = tf.add(tf.matmul(W, x_data, name="MatMul") , b , name="Add")#模型

# 代价函数 coss function 最小化方差
#loss = tf.reduce_mean(tf.square(y_model - y_data))#差值平方 均值
loss = tf.reduce_mean(tf.square(tf.sub(y_model , y_data, name="Sub"),name="Square"), name="ReduceMean")#差值平方 均值
# 优化函数  梯度下降法 学习率为0.55
#train_op = tf.train.GradientDescentOptimizer(0.55).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(0.2, name="Optimizer")
train = optimizer.minimize(loss, name="minimizer")

## 声明 summary  记录数据
# 分别记录 权重W 偏执b  以及 loss的变化  变量 tf.summary.histogram  常量 tf.summary.scalar
# summarizes = [tf.histogram_summary("W",W), tf.histogram_summary("b",b), tf.scalar_summary("loss",loss)]
summarizes = [tf.summary.histogram("W",W), tf.summary.histogram("b",b), tf.summary.scalar("loss",loss)]
# 记录操作 合并在一起
#summary_op = tf.merge_summary(summarizes)
summary_op = tf.summary.merge(summarizes)


# # you need to initialize variables (in this case just variable W)
# init = tf.initialize_all_variables()#旧的
#init =tf.global_variables_initializer()

# Launch the graph in a sessi
with tf.Session() as sess:
    #初始化变量
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()#旧版
    else:
        init = tf.global_variables_initializer()#新版
    sess.run(init)
    # 记录拟合日志
    # summary_writer = tf.train.SummaryWriter("./log/linear_regression_log", graph_def=sess.graph)
    summary_writer = tf.summary.FileWriter("./logs/linear_regression_log", graph=sess.graph)
    # 使用 tensorboard --logdir="/目录" 会给出一段网址： 打开即可
# 开始训练loop train
    for step in xrange(0, 1000):#训练1000次 
        sess.run(train)#执行优化函数
        if step % 20 == 0:#每20步 显示一下结果
            print step, sess.run(W), sess.run(b)
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=step)
# you will get the result W: [[0.100  0.200]], b: [0.300]
