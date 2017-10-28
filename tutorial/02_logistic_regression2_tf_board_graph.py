#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 逻辑回归(权重+偏置)  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

## 导入 mnist数据集 读取工具
from tensorflow.examples.tutorials.mnist import input_data

## 参数设置
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')##假数据 bool数据类型
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')##最大学习步数 整数类型
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')## 初始学习率 浮点数类型
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')## 神经元部分激活概率 浮点数类型
flags.DEFINE_string('data_dir', '../minist/MNIST_data/', 'Directory for storing data')  ## 数据集路径  字符串类型
flags.DEFINE_string('summaries_dir', './logs/mnist_LogReg_logs', 'Summaries directory')## 训练日志存储路径 字符串类型


def train():
## 输入数据
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)
  sess = tf.InteractiveSession()##交互式 会话
##创建一个多层模型 Create a multilayer model.
  ## 输入数据占位符
  with tf.name_scope('input'):## 定义操作名字
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')#输入 28*28=784维
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')#输入 10维
  ## 改变数据 形状
  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    # tf.image_summary('input', image_shaped_input, 10)#对输入数据进行 变形 并记录
    tf.summary.image('input', image_shaped_input, 10)#对输入数据进行 变形 并记录

  # 随机初始化 权重
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
  # 初始化偏置
  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  # 记录变量 的各项参数 
  # summary.scalar 记录常量 summary.histogram 记录变量
  def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):##操作名字
      mean = tf.reduce_mean(var)#均值
      tf.summary.scalar('mean/' + name, mean)#记录数据的君子
      with tf.name_scope('stddev'):#计算标准差 操作 
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('sttdev/' + name, stddev)#记录数据的标准差
      tf.summary.scalar('max/' + name, tf.reduce_max(var))# 记录数据的最大值
      tf.summary.scalar('min/' + name, tf.reduce_min(var))# 记录数据的最小值
      tf.summary.histogram(name, var)#记录变量数据本身

  ## 神经网络函数
  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """简单的神经网络层 矩阵相乘 加上偏置 使用非线性激活函数激活.
       同时也记录了一些变量的 变化
    """
    with tf.name_scope(layer_name):#操作名字 层 的名字
      # This Variable will hold the state of the weights for the layer
      #创建权重
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])#按相关尺寸 创建权重变量
        variable_summaries(weights, layer_name + '/weights')#记录变量的各项参数
      #创建偏置
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases, layer_name + '/biases')
      #创建操作
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases # y = w * x + b
        tf.summary.histogram(layer_name + '/pre_activations', preactivate)#记录变量 y
      activations = act(preactivate, 'activation')#非线性激活函数激活
      tf.summary.histogram(layer_name + '/activations', activations)#记录激活后的输出
      return activations

### 创建一个隐含层  输入 784 输出 500 名为 layer1
  hidden1 = nn_layer(x, 784, 500, 'layer1')
#### 部分神经元激活
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)#占位符
    tf.summary.scalar('dropout_keep_probability', keep_prob)#记录常量 dropout参数
    dropped = tf.nn.dropout(hidden1, keep_prob)## 部分神经元输出
## 创建输出层 输入 500 输出 10 名为 layer2  最后softmax回归输出 0~9
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.nn.softmax)

### 计算信息熵 系统混乱度
  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y) # y_ 真实标签 y预测标签
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.summary.scalar('cross entropy', cross_entropy)#记录常量信息熵
#  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
### 训练
  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)
### 计算 准确度
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      #看预测的10个标签里的最大值是否为真实标签
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)##记录准确度

  # 合并左右的 日志记录 写入相应的文件中
  merged = tf.merge_all_summaries()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                        graph=sess.graph)#外加记录 网络模型图
  test_writer  = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

  #初始化变量
  #tf.initialize_all_variables().run()
  if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
      tf.initialize_all_variables().run()#旧版
  else:
      tf.global_variables_initializer().run()#新版

  # 训练模型,记录训练日志.
  # 每隔10步, 测试 测试集 准确度,记录测试日志

## 产生训练和测试数据  喂数据
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:#训练集
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout#训练时部分神经元激活
    else:#测试集
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0#测试时 全部神经元激活
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # 每10步 记录日志 使用测试数据集
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))## 打印准确度
    else:  #99 199 ..
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # 记录日志
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()


if __name__ == '__main__':
    tf.app.run()

