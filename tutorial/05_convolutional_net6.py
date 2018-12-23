#!/usr/bin/env python
# coding: utf-8

'''
两层卷积神经网络训练手写数字识别
微信监控、调整训练过程
https://zhuanlan.zhihu.com/p/25597975?group_id=822180572054048768
https://zhuanlan.zhihu.com/p/25670072
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data  载入数据集======
from tensorflow.examples.tutorials.mnist import input_data

# Import itchat & threading
import itchat # itchat是一个开源的微信个人号接口，使用python调用微信从未如此简单。
# https://github.com/littlecodersh/ItChat
# 首先，在终端安装一下itchat包。sudo pip install itchat
# 安装完成后导入包，再登陆自己的微信。过程中会生产一个登陆二维码，扫码之后即可登陆。
'''
itchat.auto_login() 这种方法将会通过微信扫描二维码登录，
但是这种登录的方式确实短时间的登录，并不会保留登录的状态，
也就是下次登录时还是需要扫描二维码，如果加上hotReload==True,那么就会保留登录的状态，
至少在后面的几次登录过程中不会再次扫描二维码，
该参数生成一个静态文件itchat.pkl用于存储登录状态
'''
# itchatmp  开源的微信公众号、企业号接口  https://github.com/littlecodersh/itchatmp
import threading # 多线程=====

# Create a running status flag
lock = threading.Lock() # 线程锁
running = False # 程序运行标志 False未运行

# Parameters 网络参数
learning_rate = 0.001   # 学习率
training_iters = 200000 # 迭代次数
batch_size = 128        # 一次训练的数据集大小
display_step = 10       # 每隔几部打印信息

# 网络训练 闭包 函数内 定义函数======
def nn_train(wechat_name, param):
    '''
        不过首先所有print的地方都加了个itchat.send来输出日志，
        此外加了个带锁的状态量running用来做运行开关。
        此外，部分参数是通过函数参数传入的。
    '''
    # 全局变量定义 线程数据锁lock  程序运行标志running
    global lock, running
    
    # 在 程序运行标志锁Lock下 修改参数
    with lock:
        running = True

    # 读取手写字体数据集  "data/" 可以设置为已经下载好的 数据集路径，避免再次下载
    mnist = input_data.read_data_sets("data/", one_hot=True)

    # Parameters
    # learning_rate = 0.001
    # training_iters = 200000
    # batch_size = 128
    # display_step = 10
    # 解析参数
    learning_rate, training_iters, batch_size, display_step = param

    # 网络模型参数 Network Parameters
    n_input = 784  # MNIST data input (img shape: 28*28) 数据集一个样本的长度
    n_classes = 10 # MNIST total classes (0-9 digits)    输出类别 10中数字
    dropout = 0.75 # Dropout, probability to keep units  随机失活，保存存活，存活率，提高模型泛化能力

    # 图输入 tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])   # 样本坑，先占着茅坑===
    y = tf.placeholder(tf.float32, [None, n_classes]) # 标签坑，先占着茅坑===
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)# 存活率变量坑====


    # 2D卷积 + relu激活层 
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # 最大值池化层
    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                            padding='SAME')
    
    # 创建整个模型 函数
    def conv_net(x, weights, biases, dropout):
        # 1. 输入数据预处理 n*784 ---> n*28*28*1 变成二维数据，为了执行卷积运算
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # 2. 卷积 + 池化
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling) 最大值 下采样
        conv1 = maxpool2d(conv1, k=2)

        # 3. 卷积 + 池化 Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # 4.全连接层 Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1) # relu激活
        # Apply Dropout #随机失活
        fc1 = tf.nn.dropout(fc1, dropout)

        # 5. 输出层 Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input输入, 32 outputs输出，28*28 ---->下采样一次 14*14
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs       14*1 4---->下采样一次 7*7
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs 全连接层
        'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)   输出层
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # 传入参数 构建 网络模型
    pred = conv_net(x, weights, biases, keep_prob)

    # 定义损失函数 和 优化器 Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 评估模型 Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # reduce_mean() 求平均值
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # tf 初始化变量 Initializing the variables
    init = tf.global_variables_initializer()

    # 启动TF图，开启训练
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        
        # Keep training until reach max iterations
        print('Wait for lock')
        with lock:
            run_state = running # 状态
        
        print('Start')
        # 带有训练次数 和 状态标志
        while step * batch_size < training_iters and run_state:
            # 获取一个批次数据
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                        keep_prob: dropout})
            # 打印日志
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                y: batch_y,
                                                                keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))
                # 发送到 微信端=====
                itchat.send("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(acc), wechat_name)
            step += 1
            # 更新 系统状态
            with lock:
                run_state = running
        print("Optimization Finished!")
        # 训练接收  发送到客户端
        itchat.send("Optimization Finished!", wechat_name)

        # Calculate accuracy for 256 mnist test images
        # 最后进行测试
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256],
                                        keep_prob: 1.}))
        itchat.send("Testing Accuracy: %s" %
            sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                        y: mnist.test.labels[:256],
                                          keep_prob: 1.}), wechat_name)
    # 完成，修改状态标志
    with lock:
        running = False

# @语法糖 装饰器
@itchat.msg_register([itchat.content.TEXT])
def chat_trigger(msg):
    # 全局变量
    global lock, running, learning_rate, training_iters, batch_size, display_step
    '''
        如果收到微信消息，内容为『开始』，
        那就跑训练的函数（当然，为了防止阻塞，放在了另一个线程里）
    '''
    # 微信发过来的信息 开始
    if msg['Text'] == u'开始':
        print('Starting')
        with lock:
            run_state = running
        if not run_state:
            try:
                # 多线程 运行 训练工作 target=nn_train 
                threading.Thread(target=nn_train, args=(msg['FromUserName'], (learning_rate, training_iters, batch_size, display_step))).start()
            except:
                msg.reply('Running')
    # 停止命令
    elif msg['Text'] == u'停止':
        print('Stopping')
        with lock:
            running = False
    # 打印 训练参数
    elif msg['Text'] == u'参数':
        itchat.send('lr=%f, ti=%d, bs=%d, ds=%d'%(learning_rate, training_iters, batch_size, display_step),msg['FromUserName'])
    # 其他命令
    # 可以在训练开始前调整learning_rate等几个参数
    # lr=0.002
    # ds=50
    else:
        try:
            param = msg['Text'].split()
            key, value = param
            print(key, value)
            
            # 学习率
            if key == 'lr':
                learning_rate = float(value)
                
            # 训练 最大次数
            elif key == 'ti':
                training_iters = int(value)
                
            # 一次数据量
            elif key == 'bs':
                batch_size = int(value)
                
            # 显示间隔
            elif key == 'ds':
                display_step = int(value)
        except:
            pass


if __name__ == '__main__':
    # 微信客户端登录 会出现二维码 客户端扫码后即可登录
    itchat.auto_login(hotReload=True)
    # 运行程序
    itchat.run()
    
    
