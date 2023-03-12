# CycleMLP: A MLP-like Architecture for Dense Prediction

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

这是一个PaddlePaddle实现的CycleMLP。

<p align="middle">
  <img src="https://github.com/ShoufaChen/CycleMLP/blob/main/figures/teaser.png?raw=true" height="300" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/ShoufaChen/CycleMLP/blob/main/figures/flops.png?raw=true" height="300" />
</p>

**论文:** [CycleMLP: A MLP-like Architecture for Dense Prediction](https://arxiv.org/abs/2107.10224)

**参考repo:** [CycleMLP](https://github.com/ShoufaChen/CycleMLP)

项目aistudio地址：

notebook任务：https://aistudio.baidu.com/aistudio/projectdetail/3877397

在此非常感谢`ShoufaChen`贡献的[CycleMLP](https://github.com/ShoufaChen/CycleMLP)，提高了本repo复现论文的效率。


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

| 模型      | top1 acc (参考精度) | top1 acc (复现精度) | 权重 \| 训练日志 |
|:---------:|:------:|:----------:|:----------:|
| CycleMLP-B1 | 0.789 | 0.790 | checkpoint-best.pd \| train.log |

权重及训练日志下载地址：[百度网盘](https://pan.baidu.com/s/1D75OdwxWOxf9RWnzvD2ixg?pwd=oduk)


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
git clone https://github.com/flytocc/CycleMLP-paddle.git
cd CycleMLP-paddle
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

```shell
python -m paddle.distributed.launch --gpus=0,1,2,3 \
    train.py \
    /path/to/imagenet/ \
    --config configs/train/cycle_mlp/CycleMLP_B1.yaml \
    # --log_wandb --wandb_project MobileNeXt_100 \
    # --cls_label_path_train /path/to/train_list.txt \
    # --cls_label_path_val /path/to/val_list.txt \
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_dir`下train/val里的图片作为train-set/val-set。


部分训练日志如下所示。

```
[16:56:29.233819] Epoch: [261]  [ 920/1251]  eta: 0:05:50  lr: 0.000052  loss: 3.4592 (3.3812)  time: 1.0303  data: 0.0012
[16:56:49.578909] Epoch: [261]  [ 940/1251]  eta: 0:05:29  lr: 0.000052  loss: 3.7399 (3.3853)  time: 1.0171  data: 0.0015
```

### 4.2 模型评估

``` shell
python eval.py \
    /path/to/imagenet/ \
    # --cls_label_path_val /path/to/val_list.txt \
    --model CycleMLP_B1 \
    --batch_size 256 \
    --interpolation bicubic \
    --resume $TRAINED_MODEL
```
ps: 如果未指定`cls_label_path_val`，会读取`data_dir`/val里的图片作为val-set。


### 4.3 模型预测

```shell
python predict.py \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --model CycleMLP_B1 \
    --interpolation bicubic \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 211, 210, 209, 236], 'scores': [0.8659181594848633, 0.004747727885842323, 0.003118610242381692, 0.0025468438398092985, 0.0017893684562295675], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'vizsla, Hungarian pointer', 'German short-haired pointer', 'Chesapeake Bay retriever', 'Doberman, Doberman pinscher']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.8659181594848633`。

### 4.4 模型导出

```shell
python export_model.py \
    --model CycleMLP_B1 \
    --output /path/to/save/export_model/ \
    --resume $TRAINED_MODEL

python infer.py \
    --interpolation bicubic \
    --model_file /path/to/save/export_model/model.pdmodel \
    --params_file /path/to/save/export_model/model.pdiparams \
    --input_file ./demo/ILSVRC2012_val_00020010.JPEG
```


## 5. License

This project is released under MIT License.


## 6. 参考链接与文献
1. CycleMLP: A MLP-like Architecture for Dense Prediction: https://arxiv.org/abs/2107.10224
2. CycleMLP: https://github.com/ShoufaChen/CycleMLP

再次感谢`ShoufaChen`贡献的[CycleMLP](https://github.com/ShoufaChen/CycleMLP)，提高了本repo复现论文的效率。

```
@inproceedings{
chen2022cyclemlp,
title={Cycle{MLP}: A {MLP}-like Architecture for Dense Prediction},
author={Shoufa Chen and Enze Xie and Chongjian GE and Runjian Chen and Ding Liang and Ping Luo},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=NMEceG4v69Y}
}
```
