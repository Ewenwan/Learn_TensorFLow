#-*- coding:utf-8 -*-
#!/usr/bin/env python
# 两个数相乘 相加
import tensorflow as tf     # 导入tf包

a = tf.placeholder("float") # 创建一个 占位符 float类型变量   'a'
b = tf.placeholder("float") # 创建一个 占位符 float类型变量  'b'

y = tf.mul(a, b)            # 两个数相乘操作
sum1= tf.add(a, b)          # 两个数相加操作

with tf.Session() as sess: # 创建回话 用于开始一个计算
    print("%f 应该等于 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # 运行时 赋值（字典）
    print("%f 应该等于 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
    print("%f 应该等于 6.0" % sess.run(sum1, feed_dict={a: 3, b: 3}))
