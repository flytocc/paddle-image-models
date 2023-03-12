# CvT: Introducing Convolutions to Vision Transformers

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
   * [2.1 数据集](#21-数据集)
   * [2.2 复现精度](#22-复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. License](#5-license)
* [6. 参考链接与文献](#6-参考链接与文献)


## 1. 简介

这是一个PaddlePaddle实现的 CvT 。

![](https://github.com/microsoft/CvT/blob/main/figures/pipeline.svg)

**论文:** [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808)

**参考repo:** [CvT](https://github.com/microsoft/CvT)

在此非常感谢`awindsor`和`lmk123568`等人的贡献，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

### 2.1 数据集

[ImageNet](https://image-net.org/)项目是一个大型视觉数据库，用于视觉目标识别研究任务，该项目已手动标注了 1400 多万张图像。ImageNet-1k 是 ImageNet 数据集的子集，其包含 1000 个类别。训练集包含 1281167 个图像数据，验证集包含 50000 个图像数据。2010 年以来，ImageNet 项目每年举办一次图像分类竞赛，即 ImageNet 大规模视觉识别挑战赛（ILSVRC）。挑战赛使用的数据集即为 ImageNet-1k。到目前为止，ImageNet-1k 已经成为计算机视觉领域发展的最重要的数据集之一，其促进了整个计算机视觉的发展，很多计算机视觉下游任务的初始化模型都是基于该数据集训练得到的。

数据集 | 训练集大小 | 测试集大小 | 类别数 | 备注|
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
[ImageNet1k](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 |

### 2.2 复现精度

| 模型            | epochs | top1 acc (参考精度) | (复现精度) | 权重                |
|:--------------:|:------:|:------------------:|:---------:|:------------------:|
| cvt_13_224x224 |  300   | 81.6               | 81.6      | checkpoint-bext.pd |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1dhrv6DBb-LC_z3sv53ZobQ?pwd=uqch)


## 3. 准备数据与环境

### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * V100
- 框架：
  - PaddlePaddle == 2.3.2

* 安装paddlepaddle

```bash
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.3.2
# 安装CPU版本的Paddle
# pip install paddlepaddle==2.3.2
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/CvT-paddle.git
cd CvT-paddle
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

参考 [2.1 数据集](#21-数据集)，从官方下载数据后，按如下格式组织数据，即可在 PaddleClas 中使用 ImageNet1k 数据集进行训练。

```
imagenet/
    |_ train/
    |  |_ n01440764
    |  |  |_ n01440764_10026.JPEG
    |  |  |_ ...
    |  |_ ...
    |  |
    |  |_ n15075141
    |     |_ ...
    |     |_ n15075141_9993.JPEG
    |_ val/
    |  |_ ILSVRC2012_val_00000001.JPEG
    |  |_ ...
    |  |_ ILSVRC2012_val_00050000.JPEG
    |_ train_list.txt
    |_ val_list.txt
```


## 4. 开始使用

### 4.1 模型训练

* 单机多卡训练

```shell
python -m paddle.distributed.launch --gpus=0,1,2,3 \
    train.py \
    /path/to/imagenet/ \
    --config configs/train/cvt/cvt_13_224x224.yaml \
    # --log_wandb --wandb_project MobileNeXt_100 \
    # --cls_label_path_train /path/to/train_list.txt \
    # --cls_label_path_val /path/to/val_list.txt \
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_dir`下train/val里的图片作为train-set/val-set。

部分训练日志如下所示。

```
[04:32:41.592649] Epoch: [263]  [2250/2502]  eta: 0:02:21  lr: 0.000086  loss: 2.9626 (2.8445)  time: 0.5627  data: 0.0116
[04:33:09.698519] Epoch: [263]  [2300/2502]  eta: 0:01:53  lr: 0.000086  loss: 2.9557 (2.8435)  time: 0.5621  data: 0.0120
```

### 4.2 模型评估

``` shell
python eval.py \
    /path/to/imagenet/ \
    # --cls_label_path_val /path/to/val_list.txt \
    --model cvt_13_224x224 \
    --batch_size 256 \
    --interpolation bicubic \
    --resume $TRAINED_MODEL
```

ps: 如果未指定`cls_label_path_val`，会读取`data_dir`/val里的图片作为val-set。


### 4.3 模型预测

```shell
python predict.py \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --model cvt_13_224x224 \
    --interpolation bicubic \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 210, 246, 268], 'scores': [0.8321096301078796, 0.0016386241186410189, 0.0010183670092374086, 0.0009017178672365844, 0.0008596725529059768], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'German short-haired pointer', 'Great Dane', 'Mexican hairless']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8321096301078796`。

### 4.4 模型导出

```shell
python export_model.py \
    --model cvt_13_224x224 \
    --output /path/to/save/export_model/ \
    --resume $TRAINED_MODEL

python infer.py \
    --interpolation bicubic \
    --model_file /path/to/save/export_model/model.pdmodel \
    --params_file /path/to/save/export_model/model.pdiparams \
    --input_file ./demo/ILSVRC2012_val_00020010.JPEG
```

输出结果为
```
[{'class_ids': [178, 211, 210, 246, 268], 'scores': [0.8322311043739319, 0.0016395088750869036, 0.0010153381153941154, 0.00090219103731215, 0.0008562725270166993], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'German short-haired pointer', 'Great Dane', 'Mexican hairless']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8322311043739319`。与predict.py结果的误差在正常范围内。


## 5. License

This project is released under MIT License.

If you find this work or code is helpful in your research, please cite:
```
@article{wu2021cvt,
  title={Cvt: Introducing convolutions to vision transformers},
  author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},
  journal={arXiv preprint arXiv:2103.15808},
  year={2021}
}
```


## 6. 参考链接与文献

1. CvT: Introducing Convolutions to Vision Transformers: https://arxiv.org/abs/2103.15808
2. CvT: https://github.com/microsoft/CvT
