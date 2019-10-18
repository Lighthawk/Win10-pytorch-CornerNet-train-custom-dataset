# CornerNet train customed dataset on win10
Blog, win10 修改 python 配置，训练自己的数据集

## 环境与配置
win10<br>
python 3.7<br>
CornerNet_Lite<br>
Anaconda3<br>
GTX 1070<br>
Cygwin64<br>

## 吐槽
*因为最近参加天池图像处理比赛，遂从各种途径了解到超越 yolov3 的新工具 CornerNet-Lite，无奈╮(╯▽╰)╭ 几经折腾的另一系统 Ubuntu 已被玩坏，显卡驱动无法生效，安装并切换至旧内核也无解后，怒格之，空余的 1T 硬盘全部给了 Windows，并承担起这次比赛的数据仓库重任... （比赛压缩后图像200+G，1张图300~400M，这里面有点儿东西）。<br>
*于是... 我知道的，这一定是个天理不容的选择：在 windows 上做深度学习！但是，真的没办法，对于这个家里蹲、不能翻墙、显卡差点儿烧掉的电脑，我实在没有精力再重新给它折腾出一片 Ubuntu 的天地了。<br>
*就这样，作者并没有明显的说操作系统的要求，应该不像以前那样各种工具包跟 win10 水火不容吧。<br>
*实际解决时，发现只有 Ubuntu 训练自己数据集的方案，看来遇到问题要东拼西凑 + 读源码来解决了。<br>
***第一次排版让人炸毛！！！***

## 前言
（1）不讲 Python、CUDA、CUDNN、Anaconda3 的安装；<br>
（2）不讲 Cygwin64 的安装和编译；<br>
（3）不讲网络参数含义；<br>
（4）不讲自己数据转 COCO 数据集格式；<br>
（5）不讲翻墙与model下载；<br>
（6）**感谢3位大佬的链接，让我先对 Ubuntu 需要更改的地方有了一定的了解，然后再调整 win10 环境时少走了不少路，排名不分先后的三位大佬 blog：**<br>
[在SeaShips数据集上训练CenterNet网络](https://blog.csdn.net/weixin_42634342/article/details/97756458)<br>
[（绝对详细）CenterNet训练自己的数据（pytorch0.4.1）](https://blog.csdn.net/weixin_41765699/article/details/100118353)<br>
[尝试CornerNet-Lite进行目标识别并嵌入ROS](https://blog.csdn.net/qq_25349629/article/details/89493192)<br>
（7）待补；<br>

## 环境搭建过程
### （0）主要根据 [CornerNet-Lite](https://github.com/princeton-vl/CornerNet-Lite) 所说的步骤做<br>
**训练与验证之前，先更改下文件夹路径：**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/008.jpg)<br>

### （1）检查 CornerNet-Lite 安装环境<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/001.jpg)<br>
**这一步保证 Anaconda3 相关的工具包版本不要太低。原计划如果这里环境配置不合格，便 .bat 更改各工具包至 conda_packagelist.txt 版本，结果没得逞；**<br>

### （2）编译 _cpools 和 NMS<br>
**编译 Corner Pooling Layers：**<br>
```Bash
python setup.py build_ext install
```
**编译 NMS 前要修改 setup.py 文件，[参考link](https://qiita.com/sounansu/items/6836e5a4d81e157941c2)推荐翻译为英文阅读**<br>
```Python
#extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
extra_compile_args={'gcc': ['/Qstd = c99']},
```

### （3）安装 MS COCO APIs<br>
**编译步骤同 NMS，更改 `extra_compile_args`。此时可以注意到，`pycocotools` 被安装到了 `Anaconda3` 里，具体路径如下：**<br>
```Bash
 C:\ProgramData\Anaconda3\Lib\site-packages\pycocotools-2.0-py3.7-win-amd64.egg\pycocotools
```
**目录下可以看到 `coco.py` 和 `cocoeval.py`分别是 coco 数据集训练和验证的入口。我们训练自定义数据集的第一步在这里扩充 datasets。我的数据集名字为 `cancer`，则分别复制这两个 py 为 `cancer.py` 和 `cancereval.py`，文件名随意，自己区分好就OK。**<br>
在 cancer.py 中，<br>
	line 70 'class COCO:' 改为 'class CANCER:'，作为后续 `<CornerNet_Lite dir>/core/dbs` 的 `datasets` 调用时的类名<br>
	line 303， 'res = COCO()' 改为 'res = CANCER'，与 line 70 对应<br>
在 `cancereval.py` 中，<br>
	line 10, 更改类名为 'CANCEReval'，同样作为后续调用时的类名<br>

### （4）准备数据和模型<br>
**CornerNet-Lite 里后面就是将数据和模型整理好，并准备训练了。这里数据的位置和自定义更改是最麻烦的，所以，我们先放模型...模型安放很简单，丢在 `\CornerNet_Lite\cache\nnet\` 下即可，记得每个模型用模型名称的文件夹包起来。像这个样子：**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/003.jpg)<br>
**嗯？！无法翻墙怎么下载模型？推荐百度 'CornerNet 网盘'，一定找得到大佬的 orz**<br>

### （5）麻烦的放数据<br>
**我的数据集名字是 `cancer`，所以图像丢在了这个目录下 `<CornerNet_Lite dir>/data/cancer/images/`，标签在`<CornerNet_Lite dir>/data/cancer/annotations`
其中，`image/` 文件夹下继续分 `train`，`eval`，`test` 三个文件夹存放对应图像，`annotations/` 放对应的标签.json，分别为 `instances_train.json`，`instances_eval.json`，`instances_test.json`。<br>
为什么这样命名？用一张图来讲故事，应该是这样的：**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/004.jpg)<br>
**故事讲完了，出现下图就差不多可以歇了**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/009.jpg)<br>

### （6）之后还遇到了几个小插曲：<br>
**a) `"Device index must be -1 or non-negative, got -1160 "`<br>
GPU 没指定的问题，似乎是 CornerNet_Lite 默认 8 GPU并行，单 GPU 时用默认的 model config 设置， batch_size and chunk_sizes 会有分配不到 GPU的情况，设置 `batch_size=5, chunk_sizes=[5]` 两个参数一样大。[参考link](https://github.com/princeton-vl/CornerNet/issues/4) ，往下拉找大拇指**<br>
**b) 图像格式问题<br>**
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/005.jpg)<br>
	train.py 没有读到训练或验证的 image，注意图像是否放对，与，`annotations/*.json` 里是否对应；<br>
	`annotations/*.json` 里图像文件名是否正确；<br>
	讲故事的里，详细修改参数的部分，路径是否正确。<br>
**c) warning 刷屏**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/006.jpg)<br>
**顺利开始训练后，被刷满屏幕的 warning 给晃瞎了狗眼，在 train.py 里加入下面的代码，暂时屏蔽这些当前不影响训练的 warning。这是暂时的！暂时的！暂时！建议后续修改代码后运行还是打开 python 的警告，出其他问题方便定位和排查。**<br>
```Python
import warnings
warnings.filterwarnings('ignore')
```
**d) `<CornerNet_Lite dir>/core/dbs/cancer.py ` 中自定义数据集的统计参数可以自行百度查找，当前还没写这一部分，经验为0**<br>
**n) 夜深了，准备歇，初次接触 CornerNet-Lite，有太多的不熟悉的地方，欢迎大家指正错误，各个方面的都欢迎**<br>

