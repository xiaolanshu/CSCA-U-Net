# CSCA U-Net: A Channel and Space Compound Attention CNN for Medical Image Segmentation
- [CSCA U-Net: A Channel and Space Compound Attention CNN for Medical Image Segmentation](#csca-u-net--a-channel-and-space-compound-attention-cnn-for-medical-image-segmentation)
  * [1. Overview](#1-overview)
  * [2. Environment](#2-environment)
  * [3. Download Files](#3-download-files)
    + [3.1 Datasets](#31-datasets)
    + [3.2 The pth files](#32-the-pth-files)
  * [4. How run](#4-how-run)
  * [5. Acknowledge](#5-acknowledge)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>



You also can read [Chinese version](docs/README-CN.md)
## 1. Overview



## 2. Environment

you can run the following command:
```shell
conda env create -f docs/enviroment.yml
```

## 3. Download Files

### 3.1 Datasets

In this section, we provide all common datasets. 

- Polyp Dataset (Include Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, ETIS, and CVC-300. From [PraNet](https://github.com/DengPingFan/PraNet)). If you are in China, you can access `Total` to download it. If you are not in China, you can access `TrainDataset` and `TestDataset` to obtain it:
  - Total: [Aliyun](http://little-shu.com:5244/Aliyun/Datasets/Polyp%205%20Datasets.zip)
  - TrainDataset: [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing) 
  - TestDataset: [Google Drive](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing)
- 2018 Data Science Bowl: [Aliyun](http://little-shu.com:5244/Aliyun/Datasets/bowl.zip), [Google Drive](https://drive.google.com/file/d/1IWoWItLWvj1r2SbJWfBQTyPI0AngEwbb/view?usp=share_link)
- ISIC 2018 (original from [kaggle](https://www.kaggle.com/datasets/pengyizhou/isic2018segmentation/download?datasetVersionNumber=1), but I convert *.tiff to *.png): [Aliyun](http://little-shu.com:5244/Aliyun/Datasets/ISCI2018.zip), [Google Drive](https://drive.google.com/file/d/1qSNXHtV526yLLVyayOsA3bSA9LSSPBrQ/view?usp=share_link)

###  3.2 The pth files

[Google Drive](https://drive.google.com/drive/folders/1GvMXm5fehYbMFfC1mV0wHy0rHk_35JUP?usp=share_link)

## 4. How run

First, you need a pytorch environment. if you haven't, you can create a `enviroment` by [Environment](#2-Environment). 

## 5. Acknowledge

