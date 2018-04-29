## 指定目录
当前文件夹下
正确: --dataset_dir=data/quiz-w8-data
错误: --dataset_dir=/data/quiz-w8-data

## 运行
## A
>代码: python ./object_detection/eval.py
>异常: ImportError: No module named 'pycocotools'
>解决方法: 
Microsoft COCO 是 一个标注过的图片数据集，可用以目标检测、分割和描述生成等. 
Pycocotools 是 python api tools of coco...
获取源码
```
git clone https://github.com/pdollar/coco.git
```
编译
```
cd coco/PythonAPI
# install pycocotools locally
python setup.py build_ext --inplace

# install pycocotools to the Python site-packages
python setup.py build_ext install
```
## B
>代码: python ./object_detection/train.py
>异常: TypeError: `pred` must be a Tensor, a Variable, or a Python bool.
>解决方法:
搜素并打开 ssd_mobilenet_v1_feature_extractor.py
定位到 107到109行, 把 is_training=None改成 is_training=True即可 ，如下
with slim.arg_scope(
mobilenet_v1.mobilenet_v1_arg_scope(
is_training=True, regularize_depthwise=True)):
tf1.4对None的支持不友好

## C
>代码: 
>异常: 'tensorflow.contrib.data' has no attribute 'parallel_interleave'
>解决方法:
搜素并打开 dataset_util.py
定位到 132到135行, 把tf.contrib.data.parallel_interleave
改为tf.contrib.data.sloppy_interleave, 并删除sloppy=True, 如下
tf.contrib.data.sloppy_interleave(
file_read_func, 
cycle_length=config.num_readers,
block_length=config.read_block_length,))# sloppy=True))


## protoc
linux 安装protobuf,以及python版

1.下载安装包
portobuf的官网下载地址是点击打开链接: https://github.com/google/protobuf/releases，选择的版本是protobuf-all-3.5.0.tar.gz
2.解压、编译、安装
```
tar -xf  protobuf-all-3.5.0.tar.gz
cd protobuf-3.5.0
./configure
make 
make check
make install
```
3.继续安装protobuf的python模块（不需要python的，不需要安装）
```
cd ./python 
python setup.py build 
python setup.py test 
python setup.py install 
```
4.验证是否安装成功（查看安装的protobuf版本号）
```
protoc --version 
```
5.验证python模块有没有被正确安装
```
#python   
>>>import google.protobuf 
```
如果没有报错，说明安装正常

## 导入自定义模块
直接将模块放入: 安装位置/anaconda3/lib/python3.5/site-packages目录下

