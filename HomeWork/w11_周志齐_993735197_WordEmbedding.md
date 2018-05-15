
<h1>WordEmbedding<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#作业成果" data-toc-modified-id="作业成果-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>作业成果</a></span><ul class="toc-item"><li><span><a href="#词向量降维可视化--tsne.png" data-toc-modified-id="词向量降维可视化--tsne.png-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>词向量降维可视化--tsne.png</a></span></li><li><span><a href="#词向量分析与认识" data-toc-modified-id="词向量分析与认识-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>词向量分析与认识</a></span></li></ul></li><li><span><a href="#词向量简介" data-toc-modified-id="词向量简介-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>词向量简介</a></span></li><li><span><a href="#代码分析" data-toc-modified-id="代码分析-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>代码分析</a></span></li></ul></div>

# 作业成果
tinymind运行地址: https://www.tinymind.com/executions/x13vuae5

 ## 词向量降维可视化--tsne.png
 
![tsne](https://raw.githubusercontent.com/zhouzhiqi/Deep-Learning/master/HomeWork/tmp/w10_tsne.jpg)

## 词向量分析与认识

| 方框编号 | 方框内单词 | 单词语义及分析 |
| - | :- | :- |
| 1 | 二,三,四,...,十,百,千,万 | 数量词, (百,千,万)离其它有些远,隐含数量级不同 |
| 2 | 雨,雪,霜,露,白 | 代表天气, (白)可能指'白雪,白露'才被分到一起,但(白)还有颜色的意思,又离其它稍远一些 |
| 3 | 太,乐,喜,欢,歌 | 表示心情高兴,(喜,乐)(欢,歌)语义更加接近 |
| 4 | 闻,听,声 | 表声音, (闻,听)动词,更接近 |
| 5 | 舟,船 | 都是船, 旁边的(马)连同(舟,船)共指交通工具 |
| 6 | 溪,江,水 | 都是水, 小水流为(溪), 大水流为(江) |

# 词向量简介

类似于图像识别的基本过程: 预处理 -> 特征抽提 -> 分类器 -> 模式识别   
文体信息的识别也有类似的过程: 

> 1. 对于文字的预处理, 如对标点符号,非中文信息,编码问题等
> 2. 文本表示, 类似于对图像的高维特征的提取. 主要是对文本换成计算机可以处理的形式. 主要包括有One-Hot,  **`word-enbeding`**. 
> 3. 送进RNN或LSTM进行特征抽提. 由于RNN面临着梯度消失(或爆炸)的问题, LSTM被更广泛的使用
> 4. 分类器有LR,SVM,决策树等,直接选用softmax


One-Hot面临维度灾难的问题, 不常用
如今主流的方法是词向量(word-enbeding), 基本思想是: 
通过大量文本**语料库训练**, 
将每个词映射为一个**K维**的实值向量, 
通常K的取值范围从几十到几百(远远小于该语言词典的大小). 
所有这些向量构成**词向量空间**, 
而每一个向量可以认为是整个空间中的一个**点**, 
通过计算这些向量的距离进行相似度判断. 
例如, 相似的两个单词'麦克风'和'话筒', 它们在向量空间中的距离就很近. 
而两个词之间的方向更像是语义, 如'中国'同'北京'的方向和'美国'同'纽约'的方向也很相近

# 代码分析

大致步骤:   

> 1. 读入数据, 设定相关路径和参数   

> 2. 建立数据集   
> > 将数据中所有文字转成list, 并统计每个单词出现的频率;    
> > 按频率大小排序, 并给予每个单词ID, ID号越小, 代表该单词出现频率越高;    
> > 设定阈值(如5000), 并将所有低频词设为'UNK';     
> > 建立索引文件 dictionary(word -> ID), reversed_dictionary(ID -> word), 并保存     

> 3.建立并训练一个skip-gram模型, 以获得隐层权重   
> > 建立生成batch函数   
> > 构建tf.Graph()   
> > 设定相关参数并进行训练   
> > 获得词向量(即隐层权重)embeding, 并保存   

> 4.对词向量降至2维, 进行可视化处理
