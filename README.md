# CornerNet train customed dataset on win10
win10 python3.7，CornerNet-Lite 训练自己的数据集，核心是仿 COCO 数据集格式扩充自定义数据集

## 环境与配置
* CornerNet_Lite<br>
* python 3.7<br>
* Anaconda3<br>
* Cygwin64<br>
* win10<br>
* GTX 1070<br>

## 吐槽
* 最近参加天池图像处理比赛，从各种途径了解到超越 yolov3 的新工具 CornerNet-Lite，看起来很美丽，无奈╮(╯▽╰)╭ 几经折腾的 Ubuntu 已被玩坏，显卡驱动无法生效，安装并切换至旧内核也无解后，怒格之，空余的 1T 硬盘全部给了 Windows，并承担起这次比赛的数据仓库重任... （比赛压缩后图像200+G，1张图300~400M，这里面有点儿东西）。<br>
* 于是... 我知道的，这一定是个`天理不容`的选择：在 windows 上做深度学习！但是，真的没办法，对于这个家里蹲、不能翻墙、显卡差点儿烧掉的电脑，我实在没有精力再重新给它折腾出一片 Ubuntu 的天地了，抱歉。<br>
* 不过，作者并没明说操作系统的要求，应该不至于跟 win10 水火不容。网上只有 Ubuntu 训练自己数据集的方案，看来遇到问题要东拼西凑 + 读源码来解决了。<br>
* **初次接触 CornerNet-Lite，有太多的不熟悉的地方，欢迎大家指正错误，各个方面的都欢迎！！！**<br>
* **如果有帮助，可以网页右上戳下小五星 (#^.^#)**<br>
* ***第一次排版让人炸毛（气歪）！！！***

## 前言
* 不讲 Python、CUDA、CUDNN、Anaconda3 的安装<br>
* 不讲 Cygwin64 的安装和编译<br>
* 不讲网络参数含义<br>
* 不讲数据转 COCO 格式<br>
* 不讲翻墙与 model 下载<br>
* 以管理员方式运行 cmd<br>
* **感谢3位大佬的链接，让我先对 Ubuntu 需要更改的地方有一定的了解，然后再调整 win10 环境时少走了不少弯路，排名不分先后的三位大佬 blog：**<br>
	* [在SeaShips数据集上训练CenterNet网络](https://blog.csdn.net/weixin_42634342/article/details/97756458)<br>
	* [（绝对详细）CenterNet训练自己的数据（pytorch0.4.1）](https://blog.csdn.net/weixin_41765699/article/details/100118353)<br>
	* [尝试CornerNet-Lite进行目标识别并嵌入ROS](https://blog.csdn.net/qq_25349629/article/details/89493192)<br>


## 环境搭建过程
### （0）主要根据 [CornerNet-Lite](https://github.com/princeton-vl/CornerNet-Lite) 步骤做
训练与验证之前，先更改文件夹路径:
```Bash
 <CornerNet-Lite dir> --->> <CornerNet_Lite dir>
```
### （1）检查 CornerNet-Lite 安装环境
```Bash
conda create --name CornerNet_Lite --file conda_packagelist.txt --channel pytorch
source activate CornerNet_Lite
```
这一步在win10上执行没好结果，关键是保证 Anaconda3 相关的工具包版本不要太低或过高。原计划如果这环境配置不合格，就写个 bat 更改各工具包至 conda_packagelist.txt 版本，结果没得逞。<br>
```Bash
# 2019-10-18，cmd 执行，各工具包版本合适
conda update conda
conda update --all
```

### （2）编译 _cpools 和 NMS
编译 Corner Pooling Layers：<br>
```Bash
cd <CornerNet_Lite dir>\core\models\py_utils\_cpools
python setup.py build_ext install
```
编译 NMS 前要修改 setup.py 文件，[参考link](https://qiita.com/sounansu/items/6836e5a4d81e157941c2)，推荐翻译为英文阅读<br>
```Bash
cd <CornerNet_Lite dir>\core\external
make
```
```diff
# setup.py
- line 10: extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
+ line 10: extra_compile_args={'gcc': ['/Qstd = c99']},
- line 16: extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
+ line 16: extra_compile_args={'gcc': ['/Qstd = c99']},
```

### （3）安装 MS COCO APIs
```Bash
cd <CornerNet_Lite dir>\data\coco\PythonAPI
make
```
编译步骤同 NMS，make 前更改setup.py 的 `extra_compile_args`。编译后`pycocotools`被安装到了`Anaconda3`里，具体路径如下：<br>
```Bash
 C:\ProgramData\Anaconda3\Lib\site-packages\pycocotools-2.0-py3.7-win-amd64.egg\pycocotools
```
目录下`coco.py`和`cocoeval.py`分别是 coco 数据集训练和验证的入口，我们训练自定义数据集的第一步就在这里扩充 datasets。我的数据集名字为`cancer`，遂分别复制这两个 py 为`cancer.py`和`cancereval.py`，文件名随意，自己区分好就OK。<br>
```diff
# cancer.py
- line 70: 'class COCO:' 
+ line 70: 'class CANCER:'  # 作为后续 `<CornerNet_Lite dir>\core\dbs` 的 `datasets` 调用时的类名
- line 303: 'res = COCO()'
+ line 303: 'res = CANCER()'  # 与 line 70 对应
```
```diff
# cancereval.py
- line 10: 'class COCOeval:' 
+ line 10: 'class CANCEReval:'  # 同上作为后续调用时的类名
```

### （4）轻松愉快放模型
训练前模型和数据都要准备好，由于更改数据的路径和定义是最麻烦的，so，我们先放模型... 模型在[CornerNet-Lite](https://github.com/princeton-vl/CornerNet-Lite) 下载，丢在 `\CornerNet_Lite\cache\nnet\` 下即可，每个模型用同名文件夹包起来，训练时在这里读写 model：<br>
```Bash
CornerNet_Lite
│   ..  
└───cache
      └───nnet
            └───CornetNet
	    │  	    └───CornetNet_500000.pkl
	    └───CornetNet_Saccade  
	    │ 	    └───CornetNet_Saccade_500000.pkl
	    └───CornetNet_Squeeze  
		    └───CornetNet_Squeeze_500000.pkl
```
嗯？！无法翻墙怎么下载模型？百度 "CornerNet 网盘" 一定找得到大佬的 orz。<br>

### （5）脑壳生疼放数据
增加新数据集在`<CornerNet_Lite dir>\core\dbs\`里作文章，新数据集名称 `cancer`，复制`coco.py`为`cancer.py`，是增加数据集需要修改的文件，复制`detectoin.py`为`detection_cancer.py`，搭建环境时暂不调整内部参数。`__init__.py`增加新数据集索引，`cancer.py` 增加新 detection。新数据集图像放在 `<CornerNet_Lite dir>\data\cancer\images\`，标签在`<CornerNet_Lite dir>\data\cancer\annotations`。其中，`image` 文件夹下继续分 `train`，`eval`，`test` 三个文件夹存放对应图像，`annotations\` 放已转换COCO格式的标签json文件，分别为 `instances_train.json`，`instances_eval.json`，`instances_test.json`。<br>
```Bash
CornerNet_Lite
│   ..  
└───data
      └───coco  
      └───cancer
	    └───annotations
	    │     └───instances_train.json，instances_eval.json，instances_test.json
	    └───images  
		  └───train			
		  └───eval			
		  └───test
```
**...等下！为什么这样命名？用一张图来讲故事，是这样的：<br>**
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/004.jpg)<br>
**故事讲完，`<CornerNet_Lite dir>/configs/CornerNet_Squeeze.json` 根据GPU性能简单改下 `batch_size=5`和`chunk_sizes=[5]`，cmd 下运行`python train.py CornerNet`出现下图就可以稍微歇下了**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/009.jpg)<br>

### （6）几个小插曲
**a) `"Device index must be -1 or non-negative, got -1160 "`**<br>
没指定GPU，似乎是 CornerNet_Lite 单 GPU 并用默认的 model config 设置，batch_size and chunk_sizes 会有分配不到 GPU 的情况，设置`batch_size=5, chunk_sizes=[5]`两个参数一样大。[参考link](https://github.com/princeton-vl/CornerNet/issues/4) 往下拉找大拇指。<br>
**b) warning 刷屏**<br>
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/006.jpg)<br>
顺利开始训练后，被刷满屏幕的 warning 给晃瞎了狗眼，在 train.py 里加入下面的代码，暂时屏蔽这些当前不影响训练的 warning。**注意：这是暂时的！暂时的！暂时！建议后续修改代码后打开 python 的警告，出其他问题方便定位和排查。**<br>
```Python
import warnings
warnings.filterwarnings('ignore')
```
**c) `<CornerNet_Lite dir>/core/dbs/cancer.py ` 中自定义数据集的统计参数可以自行百度查找，当前还没写这一部分，经验为0**<br>
**d) 图像格式问题<br>**
![image](https://github.com/Lighthawk/CornerNet-train-win10-python/blob/master/images/005.jpg)<br>
实际是 train.py 没读入训练或验证集的图像，需要检查以下几处：
* 转COCO格式时生成的 `annotations\*.json` 内部，图像文件名是否正确；<br>
* 讲故事图，`cancer.py` line 38，`coco_dir`路径是否正确。<br>
* 讲故事图，`cancer.py` line 40~46，与`annotations\` 文件夹下的 json 文件名是否对应；<br>
