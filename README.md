# CSCA U-Net: A Channel and Space Compound Attention CNN for Medical Image Segmentation



---

## Table of Contents
- [CSCA U-Net: A Channel and Space Compound Attention CNN for Medical Image Segmentation](#csca-u-net--a-channel-and-space-compound-attention-cnn-for-medical-image-segmentation)
  * [Table of Contents](#table-of-contents)
  * [1. Overview](#1-overview)
    + [Abstract](#abstract)
  * [2. Data Files](#2-data-files)
    + [2.1 Datasets](#21-datasets)
    + [2.2 Trained weights](#22-trained-weights)
    + [2.3 Predicted Map](#23-predicted-map)
  * [3. How to run](#3-how-to-run)
    + [3.1 Create Environment](#31-create-environment)
    + [3.2 Training](#32-training)
    + [3.3 Testing and Save Pictures](#33-testing-and-save-pictures)
    + [3.4 Evaluating](#34-evaluating)
  * [4. Result](#4-result)
  * [5. Citation](#5-citation)
  * [6. Acknowledge](#6-acknowledge)
  * [7. Contact](#7-contact)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

---




**You also can read [Chinese version](docs/README-CN.md)**

## 1. Overview

![image-architecture](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/image-20221208100223315.png)

### Abstract
<p align = "justify"> 
Image segmentation is one of the vital steps in medical image analysis. A large number of methods based on convolutional neural networks have emerged, which can extract abstract features from multiple-modality medical images, learn valuable information that is difficult to recognize by humans, and obtain more reliable results than traditional image segmentation approaches. U-Net, due to its simple structure and excellent performance, is widely used in medical image segmentation. In this paper, to further improve the performance of U-Net, we propose a channel and space compound attention (CSCA) convolutional neural network, abbreviated CSCA U-Net, which increases the network depth and employs a double squeeze-and-excitation (DSE) block in the bottleneck layer to enhance feature extraction and obtain more high-level semantic features. Moreover, the characteristics of the proposed method are three-fold: (1) channel and space compound attention (CSCA) block, (2) cross-layer feature fusion (CLFF), and (3) deep supervision (DS). Extensive experiments on several available medical image datasets, including Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS, CVC-T, 2018 Data Science Bowl (2018 DSB), ISIC 2018 Challenge, and JSUAH-Cerebellum, show that CSCA U-Net achieves competitive results and significantly improves the generalization performance.
</p>

## 2. Data Files

### 2.1 Datasets

In this subsection, we provide the public data set used in the paper:

- Polyp Datasets (include Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS, and CVC-300.) \[[From PraNet](https://github.com/DengPingFan/PraNet)\]:
  - Total: \[[Aliyun](http://little-shu.com:5244/Aliyun/CSCAUNet/Datasets/Polyp%205%20Datasets.zip)\], \[[Baidu]( https://pan.baidu.com/s/1q5I2e2bbwXdW4evJdCAUpg?pwd=1111)\]
  - TrainDataset: \[[Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)\] 
  - TestDataset: \[[Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)\]
- 2018 Data Science Bowl: \[[Aliyun](http://little-shu.com:5244/Aliyun/CSCAUNet/Datasets/bowl.zip)\], \[[Baidu](https://pan.baidu.com/s/1JUzWDQydjj83GbniRgstOQ?pwd=1111)\], \[[Google Drive](https://drive.google.com/file/d/1IWoWItLWvj1r2SbJWfBQTyPI0AngEwbb/view?usp=share_link)\]
- ISIC 2018 (original from [[kaggle](https://www.kaggle.com/datasets/pengyizhou/isic2018segmentation/download?datasetVersionNumber=1)\]. I converted the images in `.tiff` format to `.png` format): \[[Aliyun](http://little-shu.com:5244/Aliyun/CSCAUNet/Datasets/ISIC2018.zip)\], \[[Baidu](https://pan.baidu.com/s/1utewXZ8Rs-X5FbTtzOy7DQ?pwd=1111)\], \[[Google Drive](https://drive.google.com/file/d/1qSNXHtV526yLLVyayOsA3bSA9LSSPBrQ/view?usp=share_link)\]

###  2.2 Trained weights

\[[Aliyun](http://little-shu.com:5244/Aliyun/CSCAUNet/snapshots.zip)\], \[[Baidu](https://pan.baidu.com/s/15QcH5fBU4uU0w-X3xu24cw?pwd=1111)\], \[[Google Drive](https://drive.google.com/drive/folders/1GvMXm5fehYbMFfC1mV0wHy0rHk_35JUP?usp=share_link)\]

### 2.3 Predicted Map

\[[Aliyun](http://little-shu.com:5244/Aliyun/CSCAUNet/Predict_map.zip)\], \[[Baidu](https://pan.baidu.com/s/1KmCXEPkAx5x1QhEx-Utypg?pwd=1111)\], \[[Google Drive](https://drive.google.com/drive/folders/1VA6J9k5XdkanpkMh4IuXe6wg0OS0lUxq?usp=sharing)\]

## 3. How to run

### 3.1 Create Environment

First of all, you need to have a `pytorch` environment, I use `pytorch 1.10`, but it should be possible to use a lower version, so you can decide for yourself.

You can also create a virtual environment using the following command (note: this virtual environment is named `pytorch`, if you already have a virtual environment with this name on your system, you will need to change `environment.yml` manually).

```shell
conda env create -f docs/enviroment.yml
```

### 3.2 Training

You can run the following command directly:

```shell
sh run.sh ### use stepLR
sh run_cos.sh ### use CosineAnnealingLR 
```

If you only want to run a single dataset, you can comment out the irrelevant parts of the `sh` file, or just type something like the following command from the command line:

```shell
python Train.py --model_name CSCAUNet --epoch 121 --batchsize 16 --trainsize 352 --train_save CSCAUNet_Kvasir_1e4_bs16_e120_s352 --lr 0.0001 --train_path $dir/data/TrainDataset --test_path $dir/data/TestDataset/Kvasir/  # you need replace ur truely Datapath to $dir.
```

### 3.3 Testing and Save Pictures

If you use a `sh` file for training, it will be tested after the training is complete.

If you use the `python` command for training, you can also comment out the training part of the `sh` file, or just type something like the following command at the command line:

```shell
python Test.py --train_save CSCAUNet_Kvasir_1e4_bs16_e120_s352 --testsize 352 --test_path $dir/data/TestDataset
```

### 3.4 Evaluating

- For evaluating the polyp dataset, you can use the `matlab` code in `eval` or use the evaluation code provided by \[[UACANet](https://github.com/plemeri/UACANet)\].
- For other datasets, you can use the code in `evaldata`.
- The reason for using a different evaluation code is to use the same methodology in the evaluation as other papers that did experiments on the dataset.

## 4. Result

- Polyp Datasets
  ![20221208101511](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/20221208101511.png)

  ![20221208101543](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/20221208101543.png)

- 2018 DSB
  ![](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/20221208101622.png)

- ISIC 2018
  ![](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/20221208101638.png)

- JSUAH-Cerebellum 
  ![20221208101600](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/20221208101600.png)

- Qualitative Results
  ![Qualitative](https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/20221208101024.png)

## 5. Citation



## 6. Acknowledge

- Many of the training strategies, datasets, and evaluation methods in this paper are based on [PraNet](https://github.com/DengPingFan/PraNet). I admire the open source spirit of Dr. Deng-Ping Fan and other authors, and am very grateful for the help provided by this work on `PraNet`.

## 7. Contact

- 📫 reach me for email:vegas_tyler@outlook.com

- <img src="https://picture-for-upload.oss-cn-beijing.aliyuncs.com/img/qq.svg" width="20" height="20">reach me for QQ: 872845991

