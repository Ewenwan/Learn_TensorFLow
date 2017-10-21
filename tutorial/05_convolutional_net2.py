#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 卷积神经网络 
# 权重(卷积核) + 偏置 AdamOptimizer
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具

batch_size = 128#每一次训练 batch大小  即每次读入128张图片进行训练
test_size = 256 #每一次测试 batch大小

## 初始化权重 #####
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

## 初始化偏置 #####
def init_bias(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

## 卷积 ##
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   #stride size=buchang=1   padding size=bianju=0  bu 0

## 池化 ##
def max_pool_2x2(x):  # 2x2 max pool img size halve
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

## 模型 ##
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
## 第一 ##
    l1a = tf.nn.relu(conv2d(X, w))      # 卷积层 l1a shape=(?, 28, 28, 32)  图像大小 28*28  32个卷积核 outputs
    l1  = max_pool_2x2(l1a)             # 池化层 l1  shape=(?, 14, 14, 32)  图像大小 14*14  32  outputs
    l1  = tf.nn.dropout(l1, p_keep_conv)# dropout 层 部分神经元激活
## 第二 ##
    l2a = tf.nn.relu(conv2d(l1, w2))    # 卷积层 l2a shape=(?, 14, 14, 64)  img size 14*14  64  outputs
    l2  = max_pool_2x2(l2a)             # 池化层 l2  shape=(?, 7, 7, 64)    img size 7*7    64  outputs
    l2  = tf.nn.dropout(l2, p_keep_conv)# dropout 层 部分神经元激活
## 第三 ##
    l3a = tf.nn.relu(conv2d(l2, w3,))   # 卷积层 l3a shape=(?, 7, 7, 128)   img size 7*7  128 outputs
    l3  = max_pool_2x2(l3a)             # 池化层 l3  shape=(?, 4, 4, 128)   img size 4*4  128 outputs
    l3  = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])#全连接  reshape to (?, 2048)   4*4*128 -> 1*2048
    l3  = tf.nn.dropout(l3, p_keep_conv)# dropout 层 部分神经元激活
## 第四 ##
    l4  = tf.nn.relu(tf.matmul(l3, w4))    # 1*2048 .* 2048*625   -> 1*625
    l4  = tf.nn.dropout(l4, p_keep_hidden) # dropout 层 部分神经元激活
## 输出层 ##
    pred_yx = tf.matmul(l4, w_o)           # 输出层 1*625  .*  625*10     ->  1*10
    return pred_yx

## 数据处理 data preprocessing
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=True)
# 训练数据 训练数据标签 测试数据  测试数据标签
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# reshape成 卷积网络需要的 图片格式
trX = trX.reshape(-1, 28, 28, 1)  # 训练数据 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 测试数据 28x28x1 input img

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
pred_yx       = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

## 计算代价函数 softmax回归 后 对数 信息增益 均值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_yx, Y))

# 优化函数   全局学习速率 0.001 
# http://blog.csdn.net/u014595019/article/details/52989301
#train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)

## 预测值
predict_op = tf.argmax(pred_yx, 1)

## 变量初始化
# init = tf.initialize_all_variables()#旧的
init =tf.global_variables_initializer()

## 启动 图 回话  Launch the graph in a session
with tf.Session() as sess:
    # 初始化变量
    #tf.initialize_all_variables().run() 旧版
    #tf.global_variables_initializer().run() 
    sess.run(init)

    for i in range(5):#训练5次
        training_batch = zip(range(0, len(trX), batch_size),   #range(start,end,scan):
                             range(batch_size, len(trX)+1, batch_size))
		# start=(0 batch_size batch_size+batch_size...) 
	        # end =(batch_size batch_size+batch_size batch_size+batch_size+batch_size ...)
        # 开始训练
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],  #0:banch_size
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})#加入 dropout参数

        test_indices = np.arange(len(teX))        # Get A Test Batch -> array[1:len(teX)]
        np.random.shuffle(test_indices)           # disorganize the Test Batch test_indices
        test_indices = test_indices[0:test_size]  # 测试数据大小

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0,## 测试时关闭 dropout
                                                         p_keep_hidden: 1.0})))#计算每次训练的正确率
# np.random.random()            ->  0~1.0
# np.random.uniform(a,b)        ->  a.0~b.0   / b.0 ~ a.0
# np.random.randint(a,b)        ->  a~b       / b   ~ a
# np.random.randrange(start, stop, step) == random.choice(range(start, stop, step) 
# np.random.choice(sequence)    ->  random select one element
# np.random.shuffle(sequence)   ->  disorganize the sequence
# np.random.sample(sequence, k) ->  random select k   element
# 



