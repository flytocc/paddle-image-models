# PeleeNet: An efficient DenseNet architecture for mobile devices

## 目录

* [1. 简介](#1-简介)
* [2. 数据集和复现精度](#2-数据集和复现精度)
* [3. 准备数据与环境](#3-准备数据与环境)
   * [3.1 准备环境](#31-准备环境)
   * [3.2 准备数据](#32-准备数据)
* [4. 开始使用](#4-开始使用)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型评估](#42-模型评估)
   * [4.3 模型预测](#43-模型预测)
   * [4.4 模型导出](#44-模型导出)
* [5. License](#7-license)
* [6. 参考链接与文献](#8-参考链接与文献)

## 1. 简介

这是一个PaddlePaddle实现的PeleeNet。

PeleeNet是一个高效的卷积神经网络（CNN）架构，由传统的卷积法构建。与其他高效架构相比，PeleeNet有很大的速度优势，可以应用于图像分类及其它的计算机视觉任务。

**论文:** [PeleeNet: An efficient DenseNet architecture for mobile devices](https://arxiv.org/pdf/1804.06882.pdf)

**参考repo:** [PeleeNet](https://github.com/Robert-JunWang/PeleeNet)

在此非常感谢`Robert-JunWang`贡献的[PeleeNet](https://github.com/Robert-JunWang/PeleeNet)，提高了本repo复现论文的效率。


## 2. 数据集和复现精度

数据集为ImageNet，训练集包含1281167张图像，验证集包含50000张图像。

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

您可以从[ImageNet 官网](https://image-net.org/)申请下载数据。

| 模型      | epochs | top1 acc (参考精度) | top1 acc (复现精度) | 权重 \| 训练日志 |
|:--------:|:------:|:--------------------------------------:|:-----:|:-----------------------------:|
| PeleeNet | 120    | - \| 0.713 (official repo) | 0.713 | 120epochs-pretrain_(checkpoint-latest.pd \| log.txt) |
| PeleeNet | 120+20 | 0.726 (paper) \| 0.716 (official repo) | 0.716 | 20epochs-finetune_(checkpoint-best.pd \| 20epochs-finetune_log.txt) |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1T0-PK7MG48qQQMwZDT5bEg?pwd=vg37)

## 3. 准备数据与环境


### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * RTX3090
- 框架：
  - PaddlePaddle >= 2.2.0

* 安装paddlepaddle

```bash
# 需要安装2.2及以上版本的Paddle，如果
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.2.0
# 安装CPU版本的Paddle
pip install paddlepaddle==2.2.0
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/PeleeNet-paddle.git
cd PeleeNet-paddle
```

* 安装requirements

```bash
pip install -r requirements.txt
```

### 3.2 准备数据

如果您已经ImageNet1k数据集，那么该步骤可以跳过，如果您没有，则可以从[ImageNet官网](https://image-net.org/download.php)申请下载。


## 4. 开始使用


### 4.1 模型训练

* 单机多卡训练

pretrain

```shell
python -m paddle.distributed.launch --gpus=0,1,2,3 \
    train.py \
    /path/to/imagenet/ \
    --config configs/train/peleenet/peleenet.yaml \
    # --log_wandb --wandb_project MobileNeXt_100 \
    # --cls_label_path_train /path/to/train_list.txt \
    # --cls_label_path_val /path/to/val_list.txt \
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_path`下train/val里的图片作为train-set/val-set。


fintune

```shell
python -m paddle.distributed.launch --gpus=0,1,2,3 \
    train.py \
    /path/to/imagenet/ \
    --config configs/train/peleenet/peleenet.yaml \
    --lr 0.005 --epochs 20 \
    --resume $PRETRAINED_MODEL --start_epoch 0 \
    # --log_wandb --wandb_project MobileNeXt_100 \
    # --cls_label_path_train /path/to/train_list.txt \
    # --cls_label_path_val /path/to/val_list.txt \
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_path`下train/val里的图片作为train-set/val-set。


部分训练日志如下所示。

```
[14:04:15.171051] Epoch: [119]  [2000/2502]  eta: 0:02:23  lr: 0.000001  loss: 1.3032 (1.2889)  time: 0.2833  data: 0.0065
[14:04:20.781305] Epoch: [119]  [2020/2502]  eta: 0:02:17  lr: 0.000001  loss: 1.3059 (1.2895)  time: 0.2794  data: 0.0118
```

### 4.2 模型评估

``` shell
python eval.py \
    /path/to/imagenet/ \
    --cls_label_path_val /path/to/val_list.txt \
    --model peleenet \
    --batch_size 256 \
    --interpolation bilinear \
    --resume $TRAINED_MODEL
```

ps: 如果未指定`cls_label_path_val`，会读取`data_path`/val里的图片作为val-set。


### 4.3 模型预测

```shell
python predict.py \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --model peleenet \
    --interpolation bilinear \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 246, 211, 236, 159], 'scores': [0.9958848357200623, 0.0028915307484567165, 0.00047466575051657856, 0.00018126785289496183, 0.00013171554019208997], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'Great Dane', 'vizsla, Hungarian pointer', 'Doberman, Doberman pinscher', 'Rhodesian ridgeback']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.9958848357200623`。

### 4.4 模型导出

```shell
python export_model.py \
    --model peleenet \
    --output /path/to/save/export_model/ \
    --resume $TRAINED_MODEL

python infer.py \
    --interpolation bilinear \
    --model_file /path/to/save/export_model/model.pdmodel \
    --params_file /path/to/save/export_model/model.pdiparams \
    --input_file ./demo/ILSVRC2012_val_00020010.JPEG
```

输出结果为
```
[{'class_ids': [178, 246, 211, 236, 159], 'scores': [0.996401309967041, 0.00265419646166265, 0.0004626315494533628, 0.00010984008986270055, 8.304142829729244e-05], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'Great Dane', 'vizsla, Hungarian pointer', 'Doberman, Doberman pinscher', 'Rhodesian ridgeback']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.996401309967041`。与predict.py结果的误差在正常范围内。


## 5. License

PeleeNet-paddle is released under `Apache License 2.0`.


## 6. 参考链接与文献

1. PeleeNet: An efficient DenseNet architecture for mobile devices: https://arxiv.org/pdf/1804.06882.pdf
2. PeleeNet: https://github.com/Robert-JunWang/PeleeNet

再次感谢`Robert-JunWang`贡献的[PeleeNet](https://github.com/Robert-JunWang/PeleeNet)，提高了本repo复现论文的效率。
```
@incollection{NIPS2018_7466,
title = {Pelee: A Real-Time Object Detection System on Mobile Devices},
author = {Wang, Robert J. and Li, Xiang and Ling, Charles X.},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {1963--1972},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7466-pelee-a-real-time-object-detection-system-on-mobile-devices.pdf}
}
```
