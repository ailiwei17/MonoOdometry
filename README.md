# **MonoOdometry** 
This repository is used for the navigation team training of the 2024 season. The repository mainly contains a simple monocular visual odometry. 

**建立本仓库的目的**

学习一些SLAM的原理

## 1. Requirements
* python 3.7+
* opencv-python> 3.4.3
* opencv-contrib-python>3.4.3
* scikit-image > 0.21.0
* open3d(还没装)

*Tips:*

推荐使用anaconda配置本仓库环境

`conda create -n monoodometry python=3.8`

`conda activate monoodometry`

## 2. Run
`python3 main.py`

目前支持的功能：

* 单目SLAM
* 局部BA优化 

[测试视频](https://tj-superpower.feishu.cn/file/XdFLbu70ao8FTZx2MSccMisFnXd)

## 3. 说明
详细的原理在代码中说明


## 4. 等待施工的项目

| 项目 | 参考 |完成|
| --- | --- | --- |
| 基础实现| [MonocularSlam](https://github.com/YunYang1994/openwork/tree/master/MonocularSlam) | 封装+移除了建图 |
| 可视化| [slam-python](https://github.com/filchy/slam-python/tree/master) |有时间再搞|
| BA优化| [pysfm](https://github.com/alexflint/pysfm) | 看不懂 |





