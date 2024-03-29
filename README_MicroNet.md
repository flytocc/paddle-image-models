# MicroNet: Improving Image Recognition with Extremely Low FLOPs

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
* [5. License](#5-license)
* [6. 参考链接与文献](#6-参考链接与文献)

## 1. 简介

这是一个PaddlePaddle实现的 MicroNet 。

**论文:** [MicroNet: Improving Image Recognition with Extremely Low FLOPs](https://arxiv.org/abs/2108.05894)

**参考repo:** [micronet](https://github.com/liyunsheng13/micronet)

在此非常感谢`liyunsheng13`、`PINTO0309`和`notplus`贡献的[micronet](https://github.com/liyunsheng13/micronet)，提高了本repo复现论文的效率。


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

| 模型         | epochs | top1 acc (参考精度) | (复现精度) |
|:-----------:|:------:|:------------------:|:---------:|
| micronet_m0 | 600    | 46.6               | 46.4      |
| micronet_m3 | 600    | 62.5               | 62.8      |

## 3. 准备数据与环境


### 3.1 准备环境

硬件和框架版本等环境的要求如下：

- 硬件：4 * V100
- 框架：
  - PaddlePaddle >= 2.3.1

* 安装paddlepaddle

```bash
# 安装GPU版本的Paddle
pip install paddlepaddle-gpu==2.3.1
# 安装CPU版本的Paddle
pip install paddlepaddle==2.3.1
```

更多安装方法可以参考：[Paddle安装指南](https://www.paddlepaddle.org.cn/)。

* 下载代码

```bash
git clone https://github.com/flytocc/MicroNet_paddle.git
cd MicroNet_paddle
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
    --config configs/train/micronet/micronet_m0.yaml \
    # --log_wandb --wandb_project MobileNeXt_100 \
    # --cls_label_path_train /path/to/train_list.txt \
    # --cls_label_path_val /path/to/val_list.txt \
```

ps: 如果未指定`cls_label_path_train`/`cls_label_path_val`，会读取`data_dir`下train/val里的图片作为train-set/val-set。


部分训练日志如下所示。

```
[14:04:15.171051] Epoch: [119]  [1000/1251]  eta: 0:02:23  lr: 0.000001  loss: 1.3032 (1.2889)  time: 0.2833  data: 0.0065
[14:04:20.781305] Epoch: [119]  [1020/1251]  eta: 0:02:17  lr: 0.000001  loss: 1.3059 (1.2895)  time: 0.2794  data: 0.0118
```

### 4.2 模型评估

``` shell
python eval.py \
    /path/to/imagenet/ \
    --cls_label_path_val /path/to/val_list.txt \
    --model micronet_m0 \
    --batch_size 256 \
    --interpolation bilinear \
    --resume $TRAINED_MODEL
```

ps: 如果未指定`cls_label_path_val`，会读取`data_dir`/val里的图片作为val-set。


### 4.3 模型预测

```shell
python predict.py \
    --infer_imgs ./demo/ILSVRC2012_val_00020010.JPEG \
    --model micronet_m0 \
    --interpolation bilinear \
    --resume $TRAINED_MODEL
```

<div align="center">
    <img src="./demo/ILSVRC2012_val_00020010.JPEG" width=300">
</div>

最终输出结果为
```
[{'class_ids': [178, 690, 176, 345, 246], 'scores': [0.7426400184631348, 0.08124781399965286, 0.0610598586499691, 0.021242130547761917, 0.015705309808254242], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'oxcart', 'Saluki, gazelle hound', 'ox', 'Great Dane']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.7426400184631348`。

### 4.4 模型导出

```shell
python export_model.py \
    --model micronet_m0 \
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
[{'class_ids': [178, 690, 176, 345, 246], 'scores': [0.7374158501625061, 0.08495301008224487, 0.06033390760421753, 0.021610060706734657, 0.016762400045990944], 'file_name': './demo/ILSVRC2012_val_00020010.JPEG', 'label_names': ['Weimaraner', 'oxcart', 'Saluki, gazelle hound', 'ox', 'Great Dane']}]
```
表示预测的类别为`Weimaraner（魏玛猎狗）`，ID是`178`，置信度为`0.7374158501625061`。与predict.py结果的误差在正常范围内。


## 5. License

MicroNet is released under MIT License.


## 6. 参考链接与文献
1. MicroNet: Improving Image Recognition with Extremely Low FLOPs: https://arxiv.org/abs/2108.05894
2. micronet: https://github.com/liyunsheng13/micronet