## Introduction
本代码仓库提出SLIBS、SLBRIN、TSUSLI和USLBRIN四个索引方法，旨在提升空间学习型索引的检索性能和更新性能，具体内容和伪代码见本人的毕业论文《增强时空特征表达的空间学习型索引》。主要工作包括四点：
1. 空间学习型索引基础框架（Basic Structure for Spatial Learned Index，SLIBS）：采用先降维、后构建单维学习型索引的空间学习型索引，对应毕业论文的第二章；
2. 空间块范围学习型索引（Spatial Learned Block Range Index，SLBRIN）：采用空间分区方法和块范围索引结构优化SLIBS，提升外存空间的检索性能，论文见[《SLBRIN: A Spatial Learned Index Based on BRIN》](https://www.mdpi.com/2220-9964/12/4/171)，对应毕业论文的第三章；
3. 时空预测学习型索引（Tempo-Spatial updatable Spatial Learned Index，TSUSLI）：采用时空序列预测算法优化SLIBS，提升更新过程的检索性能和更新性能，对应毕业论文的第四章；
4. 动态空间块范围学习型索引（Updatable Spatial Learned Block Range Index，USLBRIN）：采用TSUSLI优化SLBRIN，取长补短，并且重点解决低效模型构建导致的低效更新问题，对应毕业论文的第五章。
## Content
* data：
  * table：原始数据集，包含uniform、normal两个合成数据集和nyct真实数据集。数字1表示用于构建条件，数字2表示更新条件，10w表示测试数据集。
  * index：处理数据集，将原始数据集经过Geohash降维并排序。
  * query：检索条件，包含三种数据集的点检索条件、范围检索条件和k近邻检索条件。
  * create_data.py：数据处理代码，包含nyct数据集的数据清洗、合成数据集的生成等
* result
  * create_result*.py：结果处理代码，按照不同期刊的风格出图。
* src
  * experiment：实验部分
  * proposed_sli：所提出索引方法
  * si（Spatial Index）：所对比的空间索引
  * sli（Spatial Learned Index）：所对比的空间学习型索引
  * utils：工具类，如Geohash
  * mlp*.py：复现RMI（Recursive Model Indexes）
  * b_tree.py：复现B-tree
  * spatial_index.py：空间索引的父类
  * ts_predict.py：时空序列预测实现的预测模型
* requirement.txt
## Usage
1. 安装环境
```shell
pip install -r ./requirements.txt
```
2. 运行索引代码文件的测试用例
```shell
python ./src/si/r_tree.py
python ./src/proposed_sli/slbrin.py
```
Note:
1. ./data中只提供了10w数据量的测试数据集，完整数据集可根据[create_data.py](https://github.com/zju-niran/SLBRIN/blob/master/data/create_data.py)生成，或在留言后私发。
2. 如需切换数据集，可将代码文件中的**Distribution.NYCT_10W_SORTED**修改为**Distribution.NYCT_SORTED**或[其他数据集](https://github.com/zju-niran/SLBRIN/blob/master/src/experiment/common_utils.py)
## Copyright
Copyright © 2023, [zju-niran](https://github.com/zju-niran/), released under the [GPL-3.0 License](https://github.com/zju-niran/SLBRIN/blob/master/LICENSE).
