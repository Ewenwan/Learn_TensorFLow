# 电影评论文本进行情感分类
[参考1](https://www.cnblogs.com/lijingpeng/p/5787549.html)

[参考2](http://blog.topspeedsnail.com/archives/10399#more-10399)

      这个任务主要是对电影评论文本进行情感分类，主要分为正面评论和负面评论，
      所以是一个二分类问题，二分类模型我们可以选取一些常见的模型比如贝叶斯、逻辑回归等，
      这里挑战之一是文本内容的向量化，
      因此，我们首先尝试基于TF-IDF的向量化方法，然后尝试word2vec。
