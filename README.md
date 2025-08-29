# 复现LibraRCNN——Jittor版本
## Introduction
JDet 是一个基于[Jittor](https://github.com/Jittor/jittor)的目标检测基准库，主要聚焦于遥感图像目标检测（定向目标检测）。 
本LibrariesRCNN复现基于JDet实现，完成了论文叙述的三处优化，即:BalanceFPN、IoUBalanceSapmler与Balance_l1_loss,具体修改文件夹已标记出。

<!-- **Features**
- Automatic compilation. Our framwork is based on Jittor, which means we don't need to Manual compilation for these code with CUDA and C++.
-  -->

<!-- Framework details are avaliable in the [framework.md](docs/framework.md) -->
## Install
声明:由于本次复现完全基于JDet框架，环境配置与JDet官方完全相同。

JDet environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

注意，在部署过程中发现，Ubuntu版本似乎会影响Jittor，我使用的Ubuntu版本为22.04.3版本

## Getting Started

### Datasets
JDet 支持以下数据集，请在使用前查阅相应的文档。本次复现使用的是Dota数据集。

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

FAIR Dataset: [fair.md](docs/fair.md)

SSDD/SSDD+: [ssdd.md](docs/ssdd.md)

### Config
JDet 通过 `config-file` 定义所使用的模型、数据集以及训练/测试方法。请查阅 [config.md](docs/config.md) 以了解其工作方式。
### Train
本次复现直接在原本的FasterRCNN的基础上进行了修改
直接执行如下语句即可开始训练LibraRCNN:
```shell
python tools/run_net.py --config-file=configs/faster_rcnn_obb_r50_fpn_1x_dota.py --task=train
```

### Test
若要测试已下载的训练模型，请在配置文件的最后一行设置 `resume_path={your_checkpointspath}`。
直接执行如下语句即可开始测试LibraRCNN:
```shell
python tools/run_net.py --config-file=configs/faster_rcnn_obb_r50_fpn_1x_dota.py --task=test
```
### Test on images / Visualization
若要可视化测试模型，请执行如下语句:
```shell
python tools/run_net.py --config-file=configs/faster_rcnn_obb_r50_fpn_1x_dota.py --task=vis_test
```
用该模型训练出的可视化效果如下:
![](https://github.com/RuipengLuo/Libra_RCNN_JittorONE/blob/main/image/Snipaste_2025-08-29_17-10-00.png)

### 实验log
| 时间                     | name                                | lr                      | iter  | epoch | batch_idx | batch_size | total_loss | fps     | eta    | loss_rpn_cls | loss_rpn_bbox | rbbox_loss_cls | rbbox_acc | rbbox_loss_bbox |
|--------------------------|-------------------------------------|-------------------------|-------|-------|-----------|------------|------------|---------|--------|--------------|---------------|----------------|-----------|-----------------|
| Mon Aug 25 03:33:46 2025 | faster_rcnn_obb_r50_fpn_1x_dota     | 0.00010000000000000002  | 54300 | 11    | 4514      | 2          | 0.3273     | 8.4319  | 0:00:02| 0.0962       | 0.0273        | 0.1002         | 98.3012   | 0.1036          |

[查看完整实验日志](./work_dirs/faster_rcnn_obb_r50_fpn_1x_dota/textlog/log_2025_08_24_23_32_40.txt)


###  性能log
| 类别               | AP        |
|--------------------|-----------|
| plane              | 0.8272    |
| baseball-diamond   | 0.2373    |
| bridge             | 0.0000    |
| ground-track-field | 0.0136    |
| small-vehicle      | 0.3703    |
| large-vehicle      | 0.4870    |
| ship               | 0.3529    |
| tennis-court       | 0.7084    |
| basketball-court   | 0.0104    |
| storage-tank       | 0.4861    |
| soccer-ball-field  | 0.0844    |
| roundabout         | 0.0085    |
| harbor             | 0.2350    |
| swimming-pool      | 0.6305    |
| helicopter         | 0.0000    |
| **meanAP**         | **0.2968**|
| **iter**           | **54312** |

### loss曲线
![](https://github.com/RuipengLuo/Libra_RCNN_JittorONE/blob/main/image/Snipaste_2025-08-29_22-04-32.png)
![](https://github.com/RuipengLuo/Libra_RCNN_JittorONE/blob/main/image/Snipaste_2025-08-29_22-05-14.png)
![](https://github.com/RuipengLuo/Libra_RCNN_JittorONE/blob/main/image/Snipaste_2025-08-29_22-05-34.png)

可以看到，虽然数据量不大，但是loss函数均呈现较为明显的下降趋势

**Notice**:
训练参数含义：
1. ms: multiscale 
2. flip: random flip
3. ra: rotate aug
4. ra90: rotate aug with angle 90,180,270
5. 1x : 12 epochs
6. bc: balance category
7. mAP: mean Average Precision on DOTA1.0 test set
