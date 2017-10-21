#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 卷积神经网络 
# 仅有权重(卷积核)
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具

batch_size = 128#每一次训练 batch大小 
test_size = 256 #每一次测试 batch大小

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


## 模型 
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
 ## 第一 ## 
    # 卷积层  32个卷积核
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    # 池话层 图像大小减半
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    # droup层 神经元部分激活
    l1 = tf.nn.dropout(l1, p_keep_conv)
 ## 第二   
    # 卷积层 64个卷积核
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    # 池层层 图像大小减半
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    # droup层 神经元部分激活
    l2 = tf.nn.dropout(l2, p_keep_conv)
 ## 第三  
    # 卷积层 128个卷积核
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    # 池层层 图像大小减半
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    # 全链接层 
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    # droup层 神经元部分激活    
    l3 = tf.nn.dropout(l3, p_keep_conv)
 ## 第四   
    l4 = tf.nn.relu(tf.matmul(l3, w4)) #激活函数
    l4 = tf.nn.dropout(l4, p_keep_hidden)#droup 神经元部分激活
 ## 输出层
    pyx = tf.matmul(l4, w_o)
    return pyx
## 数据处理 data preprocessing
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# reshape成 卷积网络需要的 图片格式
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

## 训练数据的 占位符
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

### 模型参数初始化 ####
w = init_weights([3, 3, 1, 32])       # 3x3x1 卷积 conv, 32 个卷积核outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 个卷积核outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 个卷积核outputs
w4 = init_weights([128 * 4 * 4, 625]) # 全连接层 FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

## dropout参数配置  占位符 优化时再 调入
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
## 模型 
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

## 计算代价函数 softmax回归 后 对数 信息增益 均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
# 优化函数   全局学习速率 0.001 
# http://blog.csdn.net/u014595019/article/details/52989301
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
## 预测值
predict_op = tf.argmax(py_x, 1)

## 变量初始化
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

## 启动 图 回话  Launch the graph in a session
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    #tf.global_variables_initializer().run() 
    sess.run(init)

    for i in range(50):#训练50次
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
		# start=(0 batch_size batch_size+batch_size...) 
	        # end =(batch_size batch_size+batch_size batch_size+batch_size+batch_size ...)
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]#测试数据
        #测试
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0, ## 测试时关闭 dropout
                                                         p_keep_hidden: 1.0})))#计算每次训练的正确率


