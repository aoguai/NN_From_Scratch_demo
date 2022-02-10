# NN_From_Scratch_demo
一个从零开始的神经网络识别手写数字demo
## 起因
一直对AI，人工智能，深度学习方面比较感兴趣，偶然间在B站看到这样一个系列视频[《B站视频系列-从零开始的神经网络》](https://space.bilibili.com/28496477/channel/seriesdetail?sid=1267418)，于是一口气追完，边看边跟着写代码，写出了一个小demo。

## 原理
才疏学浅，暂未更新，可以去看看[原系列教程](https://space.bilibili.com/28496477/channel/seriesdetail?sid=1267418)，先占个位置，等我学透彻了再回来更新


## 源码
[原作者仓库](https://github.com/YQGong/NN_From_Scratch)的代码与实际视频教程的有出入而且功能不齐全。
在自己摸索下，写了一个相对完整的demo且有相对完整的注释，特此开源供各位参考学习
### 实现功能
+ 新建模型并训练
+ 获取最好learn_rate（学习率）
+ 模型的读入和保存
+ 读入模型并训练
+ **待更新....**

### 使用方法
首先下载解压你会得到一个

#### 目录结构
NN_From_Scratch_demo<br>
├─ MNIST<br>
│    ├─ t10k-images-idx3-ubyte.gz<br>
│    ├─ t10k-images.idx3-ubyte<br>
│    ├─ t10k-labels-idx1-ubyte.gz<br>
│    ├─ t10k-labels.idx1-ubyte<br>
│    ├─ train-images-idx3-ubyte.gz<br>
│    ├─ train-images.idx3-ubyte<br>
│    ├─ train-labels-idx1-ubyte.gz<br>
│    └─ train-labels.idx1-ubyte<br>
├─ NN_From_Scratch.py<br>
├─ main.py<br>
└─ mode<br>
  
 **MNIST文件夹** 下是MNIST 手写数字数据集<br>
 **mode文件夹** 建议存放模型（当然你也可以放在别的地方）<br>
 **NN_From_Scratch.py** 是神经网络主要代码<br>
  **main.py** 是主程序<br>
  
#### 源码介绍
 如果你也想入门的话建议从 **NN_From_Scratch.py** 部分开始浏览代码,这里实现了该demo的主要功能
 
**效果展示：**

![效果1](https://github.com/aoguai/NN_From_Scratch_demo/blob/main/images/1.png)
![效果2](https://github.com/aoguai/NN_From_Scratch_demo/blob/main/images/2.png)


## 参考
[【B站视频系列-从零开始的神经网络】](https://space.bilibili.com/28496477/channel/seriesdetail?sid=1267418)<br>
[YQGong/NN_From_Scratch:](https://github.com/YQGong/NN_From_Scratch)
