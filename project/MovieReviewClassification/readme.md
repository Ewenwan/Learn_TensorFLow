# 电影评论文本进行情感分类
[参考1](https://www.cnblogs.com/lijingpeng/p/5787549.html)

[参考2](http://blog.topspeedsnail.com/archives/10399#more-10399)

      这个任务主要是对电影评论文本进行情感分类，主要分为正面评论和负面评论，
      所以是一个二分类问题，二分类模型我们可以选取一些常见的模型比如贝叶斯、逻辑回归等，
      这里挑战之一是文本内容的向量化，
      因此，我们首先尝试基于TF-IDF的向量化方法，然后尝试 word2vec。
      
      由于处理的是字符串，我们首先要想方法把字符串转换为向量/数字表示。
      一种解决方法是可以把单词映射为数字ID。
      第二个问题是每行评论字数不同，
      而神经网络需要一致的输入(其实有些神经网络不需要，至少本帖需要)，
      这可以使用词汇表解决。
      可以使用 nltk
      
      
      
## 使用nltk  
安装nltk（自然语言工具库 Natural Language Toolkit）
$ pip install nltk
下载nltk数据：
$ python
>>> import nltk
>>> nltk.download()
测试nltk安装：
>>> from nltk.corpus import brown
>>> brown.words()
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]

> Python代码：
```python
#-*- coding:utf-8 -*-
# python3
 
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
 
import nltk
from nltk.tokenize import word_tokenize

"""
词分隔
'I'm super man'
tokenize:
['I', ''m', 'super','man' ] 
"""

from nltk.stem import WordNetLemmatizer

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""
 
pos_file = 'pos.txt'
neg_file = 'neg.txt'
 
# 创建词汇表
def create_lexicon(pos_file, neg_file):
	lex = []
	# 读取文件
	def process_file(f):
		with open(pos_file, 'r') as f:
			lex = []
			lines = f.readlines()
			#print(lines)
			for line in lines:
				words = word_tokenize(line.lower())
				lex += words
			return lex
 
	lex += process_file(pos_file)
	lex += process_file(neg_file)
	#print(len(lex))
	lemmatizer = WordNetLemmatizer()
	lex = [lemmatizer.lemmatize(word) for word in lex] # 词形还原 (cats->cat)
 
	word_count = Counter(lex)
	#print(word_count)
	# {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
	# 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
	lex = []
	for word in word_count:
		if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
			lex.append(word)        # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
	return lex
 
lex = create_lexicon(pos_file, neg_file)
#lex里保存了文本中出现过的单词。
 
# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
	dataset = []
	# lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论 
	def string_to_vector(lex, review, clf):
		words = word_tokenize(line.lower())
		lemmatizer = WordNetLemmatizer()
		words = [lemmatizer.lemmatize(word) for word in words]
 
		features = np.zeros(len(lex))
		for word in words:
			if word in lex:
				features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
		return [features, clf]
 
	with open(pos_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			one_sample = string_to_vector(lex, line, [1,0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
			dataset.append(one_sample)
	with open(neg_file, 'r') as f:
		lines = f.readlines()
		for line in lines:
			one_sample = string_to_vector(lex, line, [0,1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
			dataset.append(one_sample)
	
	#print(len(dataset))
	return dataset
 
dataset = normalize_dataset(lex)
random.shuffle(dataset)
"""
#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
"""
 
# 取样本中的10%做为测试数据
test_size = int(len(dataset) * 0.1)
 
dataset = np.array(dataset)
 
train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]
 
# Feed-Forward Neural Network
# 定义每个层有多少'神经元''
n_input_layer = len(lex)  # 输入层
 
n_layer_1 = 1000    # hide layer
n_layer_2 = 1000    # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层
 
n_output_layer = 2       # 输出层
 
# 定义待训练的神经网络
def neural_network(data):
	# 定义第一层"神经元"的权重和biases
	layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
	# 定义第二层"神经元"的权重和biases
	layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
	# 定义输出层"神经元"的权重和biases
	layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}
 
	# w·x+b
	layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
	layer_1 = tf.nn.relu(layer_1)  # 激活函数
	layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
	layer_2 = tf.nn.relu(layer_2 ) # 激活函数
	layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])
 
	return layer_output
 
# 每次使用50条数据进行训练
batch_size = 50
 
X = tf.placeholder('float', [None, len(train_dataset[0][0])]) 
#[None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')
# 使用数据训练神经网络
def train_neural_network(X, Y):
	predict = neural_network(X)
	cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001 
 
	epochs = 13
	with tf.Session() as session:
		session.run(tf.initialize_all_variables())
		epoch_loss = 0
 
		i = 0
		random.shuffle(train_dataset)
		train_x = dataset[:, 0]
		train_y = dataset[:, 1]
		for epoch in range(epochs):
			while i < len(train_x):
				start = i
				end = i + batch_size
 
				batch_x = train_x[start:end]
				batch_y = train_y[start:end]
 
				_, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x),Y:list(batch_y)})
				epoch_loss += c
				i += batch_size
 
			print(epoch, ' : ', epoch_loss)
 
		text_x = test_dataset[: ,0]
		text_y = test_dataset[:, 1]
		correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('准确率: ', accuracy.eval({X:list(text_x) , Y:list(text_y)}))
 
train_neural_network(X,Y)

"""
增加结点数量效果好了很多。
n_layer_1 = 2000 # hide layer
n_layer_2 = 2000 # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

epochs = 10
#config = tf.ConfigProto(log_device_placement=True)

# config=tf.ConfigProto(log_device_placement=True)
with tf.device(“/gpu:0”):
with tf.Session() as session:
session.run(tf.global_variables_initializer())
epoch_loss = 0

i = 0
random.shuffle(train_dataset)
train_x = dataset[:, 0]
train_y = dataset[:, 1]
for epoch in range(epochs):
while i < len(train_x):
start = i
end = i + batch_size

batch_x = train_x[start:end]
batch_y = train_y[start:end]

_, c = session.run([optimizer, cost_func], feed_dict={X: list(batch_x), Y: list(batch_y)})
epoch_loss += c
i += batch_size

print(epoch, ' : ', epoch_loss)

text_x = test_dataset[:, 0]
text_y = test_dataset[:, 1]
correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
print('准确率: ', accuracy.eval({X: list(text_x), Y: list(text_y)}))
"""
```

