#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 自编码 
# AutoEncoder大致是一个将数据的高维特征进行压缩降维编码，
# 再经过相反的解码过程的一种学习方法。学习过程中通过解码得到的最终结果
# 与原数据进行比较，通过修正权重偏置参数降低损失函数，不断提高对原数据的复原能力。
# 学习完成后，前半段的编码过程得到结果即可代表原数据的低维“特征值”。
# 通过学习得到的自编码器模型可以实现将高维数据压缩至所期望的维度，原理与PCA相似。
# http://blog.csdn.net/marsjhao/article/details/68950697
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  
  
# 导入MNIST数据  
from tensorflow.examples.tutorials.mnist import input_data# 读取mnist手写字体数据工具
mnist = input_data.read_data_sets("../minist/MNIST_data/", one_hot=False)  

## 参数设置 ## 
learning_rate = 0.01 #学习率 
training_epochs = 10 #训练次数 
batch_size = 256  # 一次训练输入的图片数量
display_step = 1  # 可视化步长
examples_to_show = 10#显示10张
n_input = 784 #输入维度  28*28 
  
# 输入数据占位符  
X = tf.placeholder("float", [None, n_input])  
  
# 用字典的方式存储各隐藏层的参数  
n_hidden_1 = 256 # 第一编码层神经元个数  
n_hidden_2 = 128 # 第二编码层神经元个数  
# 权重和偏置的变化在编码层和解码层顺序是相逆的  
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数  
weights = {  
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),  
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),  
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),  
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),  
}  
biases = {  
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),  
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),  
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),  
}  
  
# 每一层结构都是 xW + b  
# 构建编码器  
def encoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                                   biases['encoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                                   biases['encoder_b2']))  
    return layer_2  
  
  
# 构建解码器  
def decoder(x):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),  
                                   biases['decoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),  
                                   biases['decoder_b2']))  
    return layer_2  
  
# 构建模型  
encoder_op = encoder(X)  
decoder_op = decoder(encoder_op)  
  
# 预测  
y_pred = decoder_op  
y_true = X  
  
# 定义代价函数和优化器  
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) #最小二乘法  差平方均值
# 优化函数   全局学习速率 0.01 
# http://blog.csdn.net/u014595019/article/details/52989301
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  
  
with tf.Session() as sess:  
    # tf.initialize_all_variables() no long valid from  
    # 2017-03-02 if using tensorflow >= 0.12  
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  
        init = tf.initialize_all_variables()  
    else:  
        init = tf.global_variables_initializer()  
    sess.run(init)  
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  
    total_batch = int(mnist.train.num_examples/batch_size) #总批数  
    for epoch in range(training_epochs):#训练次数  
        for i in range(total_batch):# 每次训练遍历训练集所需次数 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0  
            # Run optimization op (backprop) and cost op (to get loss value)  
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})  
        if epoch % display_step == 0: #每隔 多少步显示一次 
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))  
    print("Optimization Finished!")  
  ## 画图显示 编码前后的图像
    encode_decode = sess.run(  
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})  
    f, a = plt.subplots(2, 10, figsize=(10, 2))  
    for i in range(examples_to_show):  
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))#原来的数据
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))    #编码后的数据
    plt.show() 

