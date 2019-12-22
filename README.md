# CS598 Term Project

## 3D Point Cloud Segmentation and Classification


This is the repo aims to develop the implementation code for the CS598 project.

Authors:

  * [Jiayi Luo](https://github.com/luojy95/)
  * [Zhoutong Jiang](https://github.com/timilan/)
  * [Zixu Zhao](https://github.com/)

Some useful resources:
> [PointNet original implementation](https://github.com/charlesq34/pointnet)
> 
> [PointNet pytorch version implementation](https://github.com/fxia22/pointnet.pytorch)

> [PointNet++ original implementation](https://github.com/charlesq34/pointnet2)
> 
> [PointNet Paper](https://arxiv.org/pdf/1612.00593.pdf)
> 
> [PointNet++ Paper](https://arxiv.org/pdf/1706.02413.pdf)


Dataset:
-------------
This repo contains some sample preprocessed Lyft data

  * [ShapeNet](https://www.shapenet.org/)
  * [LyftDataset](https://level5.lyft.com/dataset/)


Usage:
-------------

    1. Down Load either dataset
    2. If using LyftDataset, run utils/pre_process_LyftDataset.py
    3. Run the model/train_classification.py or model/train_segmentation.py, remember to set the preprocess tag to True at the first time.
    4. The model will be saved at the same directory as the script



