#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 卷积神经网络  tensorboard 显示优化记录
# 权重(卷积核) + 偏置 AdamOptimizer
# https://github.com/ahangchen/GDLnotes
from __future__ import print_function
import tensorflow as tf
## 导入 mnist数据集 读取工具
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
mnist_path = 'mnist'

import os#系统操作
import cPickle as pickle#模型的保存与载入  将内存中的对象转换成为文本流  cPickle速度快 于 Pickle 1000倍
'''
#open(路径+文件名,读写模式)
#读写模式:r只读,r+读写,w新建(会覆盖原有文件),a追加,b二进制文件.常用模式
rU 或 Ua 以读方式打开, 同时提供通用换行符支持 (PEP 278)
w     以写方式打开，
a     以追加模式打开 (从 EOF 开始, 必要时创建新文件)
r+     以读写模式打开
w+     以读写模式打开 (参见 w )
a+     以读写模式打开 (参见 a )
rb     以二进制读模式打开
wb     以二进制写模式打开 (参见 w )
ab     以二进制追加模式打开 (参见 a )
rb+    以二进制读写模式打开 (参见 r+ )
wb+    以二进制读写模式打开 (参见 w+ )
ab+    以二进制读写模式打开 (参见 a+ )
注意：
1、使用'W'，文件若存在，首先要清空，然后（重新）创建，
2、使用'a'模式 ，把所有要写入文件的数据都追加到文件的末尾，
   即使你使用了seek（）指向文件的其他地方，如果文件不存在，将自动被创建。
'''
## 保存模型文件
def save_obj(pickle_file, obj):
    try:
        f = open(pickle_file, 'wb')# 以二进制写模式打开
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)# 写入文件
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)# 打印大小信息
## 载入模型文件 到 内存
def load_pickle(pickle_name):
    if os.path.exists(pickle_name):
        return pickle.load(open(pickle_name, "r"))#只读方式打开
    return None

# 最大值池化
def maxpool2d(data, k=2, s=2):   # 和尺寸            步长   (原尺寸-k + pading)/s
    return tf.nn.max_pool(data, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding='SAME')
# 改变数据形状 1*784  ————>   28*28 
def img_reshape(data, length):
    img_size = 28
    depth = 1
    return np.array(data).reshape(length, img_size, img_size, depth)
# 改变标签形状   ——————>  1*10
def label_reshape(data, length):
    label_size = 10
    return np.array(data).reshape(length, label_size)

## 格式化 手写字体数据集 
def format_mnist():
    mnist = load_pickle(mnist_path)## 载入格式化后的数据
    if mnist is None:## 如果没有重新提取
        mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)## 读取原始数据
        # save_obj(mnist_path, mnist)## 保存数据  这里为了提高读书数据集的速度保存读取好的数据 但是内存占用 还是不保存了
    train_length = len(mnist.train.labels)## 训练数据集大小
    valid_length = len(mnist.validation.labels)##验证数据集大小
    test_length = len(mnist.test.labels)## 测试数据集大小
    return img_reshape(mnist.train.images, train_length), label_reshape(mnist.train.labels, train_length), \
           img_reshape(mnist.validation.images, valid_length), label_reshape(mnist.validation.labels, valid_length), \
           img_reshape(mnist.test.images, test_length), label_reshape(mnist.test.labels, test_length)


###### 记录变量 的各项参数：均值 标准差 最大值 最小值 数据本身#####
# summary.scalar 记录常量 summary.histogram 记录变量
## 老版本
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):##操作名字
        mean = tf.reduce_mean(var)#均值
        tf.scalar_summary('mean/' + name, mean)#记录数据的均值
        with tf.name_scope('stddev'):#计算标准差 操作 
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)#记录数据的标准差
        tf.scalar_summary('max/' + name, tf.reduce_max(var))# 记录数据的最大值
        tf.scalar_summary('min/' + name, tf.reduce_min(var))# 记录数据的最小值
        tf.histogram_summary(name, var)#记录变量数据本身
#新版本
def variable_summary(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def large_data_size(data):
    return data.get_shape()[1] > 1 and data.get_shape()[2] > 1

## 卷积神经网络 训练步骤  stride_ps 步长
def conv_train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels, image_size,
               num_labels, basic_hps, stride_ps, drop=False, lrd=False, get_grad=False, norm_list=None):
    ##  每次使用的数据个数
    batch_size = basic_hps['batch_size'] # 一次训练数据的大小 一次训练的图片数量
    ## 卷积核尺寸相关
    patch_size = basic_hps['patch_size'] # 隐含层 卷积核尺寸相关
    ## 卷积核数量相关
    depth = basic_hps['depth']# 输入层 卷积核数量  depth * (i + 1)  i=0,1,2,...每一层 卷积核数量
    ## 输出层 全连接层神经元数量
    first_hidden_num = 192    # 输出层 第一全连接层 神经元数量
    second_hidden_num = basic_hps['num_hidden']# 输出层 第二全连接层 神经元数量
    num_channels = 1#图像的通道数  灰度图像为1  彩色图像为3
    layer_cnt = basic_hps['layer_sum']##卷积层总层数 包含输入层卷积层 和 隐含层卷积层
    loss_collect = list()

    graph = tf.Graph()
    with graph.as_default():
     ##### 训练输入数据和标签 占位符.
        with tf.name_scope('input'):## 操作名称 在tf图中 的名字 输入input操作
            with tf.name_scope('data'):#数据 #batch_size 每次训练 使用的图片数量
                tf_train_dataset = tf.placeholder(
                    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
                variable_summary(tf_train_dataset)#记录训练数据集
            with tf.name_scope('label'):#标签
                tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
                variable_summary(tf_train_labels)#记录训练数据集的标签
#### 初始化权重和偏置 ####
     #### 初始化输入卷积层 权重和偏置. # the third parameter must be same as the last layer depth
        with tf.name_scope('input_cnn_filter'):## 操作名称 在tf图中 的名字 输入卷积核
            with tf.name_scope('input_weight'):# 输入层卷积核 权重 随机初始化 不要是0
                input_weights = tf.Variable(tf.truncated_normal(
                    # 卷积核尺寸 patch_size, patch_size  num_channels 数据输入层 通道数量   depth 卷积核数量 即为输入的通道数
                    [patch_size, patch_size, num_channels, depth], stddev=0.1), name='input_weight')
                variable_summary(input_weights)#记录数据
            with tf.name_scope('input_biases'):# 输入层偏置 与本次卷积核数量depth一致
                input_biases = tf.Variable(tf.zeros([depth]), name='input_biases')
                variable_summary(input_weights)#记录数据
     #### 初始化中间隐含卷积网络层 权重和偏置
        mid_layer_cnt = layer_cnt - 1#当前剩余层数
        layer_weights = list()
        # 输入层 卷积核数量 depth * (0 + 1) 第一层隐含层卷积核数量depth * (0 + 2) ,第二层 depth * (1 + 2)...
        # 对于偏置 数量 depth×1 为 input_biases的大小，第一层隐含层偏置数量 depth * (0 + 2)，第二层 depth * (1 + 2)...
        layer_biases = [tf.Variable(tf.constant(1.0, shape=[depth * (i + 2)])) for i in range(mid_layer_cnt)]
        for i in range(mid_layer_cnt):
            variable_summary(layer_biases)

     ###### 初始化 输出层中的 两层全连接层 权重和参数####
# [shapes[0],first_hidden_num]*[first_hidden_num, second_hidden_num] * [second_hidden_num, num_labels] ——> [shapes[0], num_labels]
        output_weights = list()
        output_biases = tf.Variable(tf.constant(1.0, shape=[first_hidden_num]))##输出权重尺寸 为了匹配 输出全连接层
        with tf.name_scope('first_nn'):#### 输出层第一层 全连接层####
            with tf.name_scope('weights'): #权重
                first_nn_weights = tf.Variable(tf.truncated_normal(
                    [first_hidden_num, second_hidden_num], stddev=0.1))
                variable_summary(first_nn_weights)
            with tf.name_scope('biases'): #偏置
                first_nn_biases = tf.Variable(tf.constant(1.0, shape=[second_hidden_num]))
                variable_summary(first_nn_weights)
        with tf.name_scope('second_nn'):#### 输出层第二层 全连接层####
            with tf.name_scope('weights'):#权重
                second_nn_weights = tf.Variable(tf.truncated_normal(
                    [second_hidden_num, num_labels], stddev=0.1))
                variable_summary(second_nn_weights)
            with tf.name_scope('biases'):#偏置
                second_nn_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
                variable_summary(second_nn_biases)

   ###################################################
   ###### 定义模型.
        def model(data, init=True):
            if not large_data_size(data) or not large_data_size(input_weights):
                stride_ps[0] = [1, 1, 1, 1] # 不是 1*784 类型 而是 28*28 图像类型
          ###############################################################
          ### 第一层 输入层 卷积层####
            # 卷积层
            with tf.name_scope('first_cnn'):## 输入层卷积层 数据 卷积核 步长
                conv = tf.nn.conv2d(data, input_weights, stride_ps[0], use_cudnn_on_gpu=True, padding='SAME')
                if init:
                    print('init')
                    variable_summary(conv)# 记录卷积层输出
            # 池化层
            with tf.name_scope('first_max_pool'):#### 最大值池化层
                conv = maxpool2d(conv)
                if init:
                    variable_summary(conv)# 记录池化层输出
            # 非线性激活层
            hidden = tf.nn.relu6(conv + input_biases)#加上偏置 后非线性激活
            if init:
                tf.summary.histogram('first_act', hidden)#记录非线性激活层输出
            # 部分神经元激活层
            if drop and init:
                with tf.name_scope('first_drop'):
                    hidden = tf.nn.dropout(hidden, 0.8, name='drop1')## 0.8概率激活神经元
                    tf.summary.histogram('first_drop', hidden)#记录 部分激活层输出
           ###################################################################
           ### 后续多层隐含卷积层 ####
            for i in range(mid_layer_cnt):
                with tf.name_scope('cnn{i}'.format(i=i)):
                 ##### 初始化每一层 权重和偏置 ####
                    if init:
                        hid_shape = hidden.get_shape()#输入数据的形状
                        # print(hid_shape)
                        filter_w = patch_size / (i + 1)#卷积核 宽
                        filter_h = patch_size / (i + 1)#卷积核 高
                        # print(filter_w)
                        # print(filter_h)
                        # 避免卷积核 尺寸大于输出数据的尺寸
                        if filter_w > hid_shape[1]:
                            filter_w = int(hid_shape[1])
                        if filter_h > hid_shape[2]:
                            filter_h = int(hid_shape[2])
                        with tf.name_scope('weight'):# 随机初始化 卷积核  卷积核尺寸 filter_w, filter_h
                                                     # 上一层通道数depth * (i + 1) 即本层的输入通道数
                                                     # 本层卷积核数量depth * (i + 2) ，即本层输出通道数 ，即下一层输入通道数
                            layer_weight = tf.Variable(tf.truncated_normal(
                                shape=[filter_w, filter_h, depth * (i + 1), depth * (i + 2)], stddev=0.1))
                            variable_summary(layer_weight)
                        layer_weights.append(layer_weight)#记录所有层的权重
                 ####### 构建每一层网络 ###
                    if not large_data_size(hidden) or not large_data_size(layer_weights[i]):
                        # print("is not large data")
                        stride_ps[i + 1] = [1, 1, 1, 1]
                    # print(stride_ps[i + 1])
                    # print(len(stride_ps))
                    # print(i + 1)
                    ### 卷积层 ###
                    with tf.name_scope('conv2d'):
                        conv = tf.nn.conv2d(hidden, layer_weights[i], stride_ps[i + 1], use_cudnn_on_gpu=True, padding='SAME')
                        if init:
                            variable_summary(conv)
                    #### 池化层 ###
                    with tf.name_scope('maxpool2d'):
                        if not large_data_size(conv):#输入数据尺寸过小
                            print('not large')
                            conv = maxpool2d(conv, 1, 1)## 卷积核为1  步长为1
                            if init:
                                variable_summary(conv)
                        else:
                            conv = maxpool2d(conv)#最大值池化 尺寸变为一般
                            if init:
                                variable_summary(conv)
                    #### 非线性激活层 ###
                    with tf.name_scope('act'):
                        hidden = tf.nn.relu6(conv + layer_biases[i])
                        if init:
                            variable_summary(conv)
          ###################################################################
          ###### 输出层 全连接层部分 ###
            shapes = hidden.get_shape().as_list() #隐含层输出数据的尺寸
            shape_mul = 1##全连接
            for s in shapes[1:]:
                shape_mul *= s
          #输出层 缓冲层 权重参数初始化
            ## 隐含层 过渡到 输出 全连接层 的过渡层  shapes[0]为一次输入训练的图片数量
            ##  [shapes[0], 1] * [1 , first_hidden_num] ----> [shapes[0],first_hidden_num]
            if init:
                with tf.name_scope('output'):
                    output_size = shape_mul
                    with tf.name_scope('weights'):
                        output_weights.append(tf.Variable(tf.truncated_normal([output_size, first_hidden_num], stddev=0.1)))
                        variable_summary(output_weights)
            # 变形 [shapes[0], 1]
            reshape = tf.reshape(hidden, [shapes[0], shape_mul])## reshape 成一维列向量  全连接  类似 线性回归/逻辑回归
            # 乘权重+偏置 在非线性激活 
            # [shapes[0], 1] * [1 , first_hidden_num] ----> [shapes[0],first_hidden_num]
            with tf.name_scope('output_act'):
                hidden = tf.nn.relu6(tf.matmul(reshape, output_weights[0]) + output_biases)
                if init:
                    tf.summary.histogram('output_act', hidden)
            # 部分神经元激活            
            if drop and init:
                with tf.name_scope('output_drop'):
                    hidden = tf.nn.dropout(hidden, 0.5)
                    tf.summary.histogram('output_drop', hidden)

            ##输出层 第一层 全连接层
            with tf.name_scope('output_wx_b'):
                hidden = tf.matmul(hidden, first_nn_weights) + first_nn_biases
                if init:
                    tf.summary.histogram('output_wx_b', hidden)
            if drop and init:# 部分神经元激活
                with tf.name_scope('final_drop'):
                    hidden = tf.nn.dropout(hidden, 0.5)
                    tf.summary.histogram('final_drop', hidden)
            ##输出层 第二层 全连接层
            with tf.name_scope('final_wx_b'):
                hidden = tf.matmul(hidden, second_nn_weights) + second_nn_biases
                if init:
                    tf.summary.histogram('final_wx_b', hidden)
            return hidden

##########################################
       #### 模型输出.###############
        with tf.name_scope('logits'):
            logits = model(tf_train_dataset)
            tf.summary.histogram('logits', logits)
       #### 模型误差 loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
            tf.summary.histogram('loss', loss)

        # 训练步骤的 优化器 Optimizer.
        with tf.name_scope('train'):
            if lrd:
                cur_step = tf.Variable(0)   # 优化步数计数变量 
                starter_learning_rate = 0.06# 开始学习率
                # 随着学习步数增加 学习率按指数 递减
                learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 600, 0.1, staircase=True)
                # 随机梯度下降优化器
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=cur_step)
            else:
                # Adagrad 优化器 
                optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

        # 训练 验证 测试时的 预测结果 使用softmax回归.
        with tf.name_scope('train_predict'):
            #train_prediction = tf.nn.softmax(logits)  # train_prediction 为32*10  32为一次训练输入的图片数量 每个图片预测一个1*10的类别概率
        #    with sess.as_default():
            train_prediction = tf.argmax(logits, 1) #每一行为 一个图片 预测的10类 概率其中最大的一个即为预测的标签
            #variable_summary(train_prediction)
        # with tf.name_scope('valid_predict'):
        #     valid_prediction = tf.nn.softmax(model(tf_valid_dataset, init=False))
        #     variable_summary(valid_prediction, 'valid_predict')
        # with tf.name_scope('test_predict'):
        #     test_prediction = tf.nn.softmax(model(tf_test_dataset, init=False))
        #     variable_summary(test_prediction, 'test_predict')
        merged = tf.summary.merge_all()## 记录所有 参数

##### 记录日志标志 ######################################
    summary_flag = True## 记录日志标志
##### 日志保存路径 ######################################
    summary_dir = 'logs/cnn_mnist'##日志保存路径
    # 使用 tensorboard --logdir="/目录" 会给出一段网址： 打开即可
    if tf.gfile.Exists(summary_dir):#如果存在目录
        tf.gfile.DeleteRecursively(summary_dir)#删除原来的
    tf.gfile.MakeDirs(summary_dir)#如果不存在，则新建目录

    num_steps = 1001#总的训练步数
    #'''
    with tf.Session(graph=graph) as session:
        train_writer = tf.summary.FileWriter(summary_dir + '/train', session.graph)##训练时记录参数优化记录 以及保存 模型结构图文件
        valid_writer = tf.summary.FileWriter(summary_dir + '/valid')
        ## 初始化变量
	if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
	    tf.initialize_all_variables().run()#旧版
	else:
	    tf.global_variables_initializer().run()#新版
        print('Initialized')## 打印初始化结束信息
        mean_loss = 0
        print('running')## 开始优化
        for step in range(num_steps):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            ### 准备每次的训练数据
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)#每次的偏移量
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            ## 启动回话
            if summary_flag:
                summary, _, l, predictions = session.run(
                    [merged, optimizer, loss, train_prediction], options=run_options, feed_dict=feed_dict)
            else:
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], options=run_options, feed_dict=feed_dict)
            mean_loss += l
            if step % 5 == 0:#每5次 计算平均LOSS
                mean_loss /= 5.0
                loss_collect.append(mean_loss)## 记录 平均loss
                mean_loss = 0
                if step % 50 == 0:#每50次
                    print('%d , loss:     %f' % (step, l))#打印LOSS
                    #print (tf.convert_to_tensor(np.argmax(batch_labels, axis=1)))
                    print (np.argmax(batch_labels, axis=1))#32个真实标签
                    print (session.run(train_prediction, options=run_options, feed_dict=feed_dict))#32个预测标签
                    print('%d , train_accuracy: %.4f' % (step, np.mean(np.argmax(batch_labels, axis=1) == session.run(train_prediction, options=run_options, feed_dict=feed_dict))))
                    if step % 100 == 0 and summary_flag:
                        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                        train_writer.add_summary(summary, step)
                        print('Adding run metadata for', step)
                if summary_flag:
                    #test_indices = np.arange(len(teX)) # Get A Test Batch  valid_dataset, valid_labels
                    #np.random.shuffle(test_indices)
                    #test_indices = test_indices[0:test_size]#测试数据
		    ### 准备每次的验证数据###
		    offset = (step * batch_size) % (valid_labels.shape[0] - batch_size)#每次的偏移量
		    batch_data_valid = valid_dataset[offset:(offset + batch_size), :, :, :]
		    batch_labels_valid = valid_labels[offset:(offset + batch_size), :]
		    feed_dict_valid = {tf_train_dataset: batch_data_valid, tf_train_labels: batch_labels_valid}
                    print('%d , valid_accuracy: %.4f' % (step, np.mean(np.argmax(batch_labels_valid, axis=1) == session.run(train_prediction, options=run_options, feed_dict=feed_dict_valid))))
                    #valid_writer.add_summary(summary, step)
            if summary_flag:
                train_writer.add_summary(summary, step)
        train_writer.close()
        valid_writer.close()
        # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    #'''
def hp_train():
    image_size = 28#照片大小 28*28
    num_labels = 10#类别数量 0~9
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = \
        format_mnist()
    pick_size = 2048#
    valid_dataset = valid_dataset[0: pick_size, :, :, :]
    valid_labels = valid_labels[0: pick_size, :]
    test_dataset = test_dataset[0: pick_size, :, :, :]
    test_labels = test_labels[0: pick_size, :]
    basic_hypers = {
        'batch_size': 32,#每次训练32张图片
        'patch_size': 5, #卷积核尺寸           patch_size / (i + 1)
        'depth': 16,     #卷积核起始数量为16   depth * (i + 1)
        'num_hidden': 64,#输出层 第二全连接层  最后一层神经元数量  6*  64*10 ----> 1*10
        'layer_sum': 2   #卷积层总层数 中间隐含卷层数为 layer_sum-1
    }
    stride_params = [[1, 2, 2, 1] for _ in range(basic_hypers['layer_sum'])]
    conv_train(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset,
               test_labels,
               image_size, num_labels, basic_hypers, stride_params, drop=True, lrd=False)


if __name__ == '__main__':
    hp_train()

