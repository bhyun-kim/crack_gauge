# DeepCrack

> [DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation](https://www.sciencedirect.com/science/article/pii/S0925231219300566)

## Introduction

[Official Repo](https://github.com/yhlleo/DeepSegmentor)

## Abstract

Automatic crack detection from images of various scenes is a useful and challenging task in practice. In this paper, we propose a deep hierarchical convolutional [neural network](https://www.sciencedirect.com/topics/neuroscience/neural-network "Learn more about neural network from ScienceDirect's AI-generated Topic Pages") (CNN), called as DeepCrack, to predict pixel-wise crack segmentation in an end-to-end method. DeepCrack consists of the extended Fully [Convolutional Networks](https://www.sciencedirect.com/topics/computer-science/convolutional-network "Learn more about Convolutional Networks from ScienceDirect's AI-generated Topic Pages") (FCN) and the Deeply-Supervised Nets (DSN). During the training, the elaborately designed model learns and aggregates multi-scale and multi-level features from the low [convolutional layers](https://www.sciencedirect.com/topics/computer-science/convolutional-layer "Learn more about convolutional layers from ScienceDirect's AI-generated Topic Pages") to the high-level convolutional layers, which is different from the standard approaches of only using the last convolutional layer. DSN provides integrated direct supervision for features of each convolutional stage. We apply both guided filtering and [Conditional Random Fields](https://www.sciencedirect.com/topics/computer-science/conditional-random-field "Learn more about Conditional Random Fields from ScienceDirect's AI-generated Topic Pages") (CRFs) methods to refine the final prediction results. A benchmark dataset consisting of 537 images with manual annotation maps are built to verify the effectiveness of our proposed method. Our method achieved state-of-the-art performances on the proposed dataset (mean I/U of 85.9, best F-score of 86.5, and 0.1 s per image).

## Results and Models

### DeepCrack


| Method | Backbone | Crop<br />Size | Lr schd | Mem<br />(GB) | Inf<br />time<br />(fps) | Device | mIoU  | config             | download                                                                                                                                                                                                                                                                                                                                                         |
| :------: | ---------- | :--------------- | --------- | --------------- | -------------------------- | -------- | ------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CGNet | M3N21    | 1024×1024     | 20000   | -             | -                        | A100   | 86.59 | [config]([https://](https://github.com/bhyun-kim/crack_gauge/blob/main/configs/deep_crack/cgent_1024x1024.py)) | [model](https://www.dropbox.com/scl/fi/9j4nkxf0u7yg6pmy3ddq0/iter_14000.pth?rlkey=m7a3q3g32wdsdc1e74rsxjq3c&dl=0https:/), [log](https://www.dropbox.com/scl/fi/zyv3fggf8wgwqvcxj97ry/20230811_223219.log?rlkey=eh3z0rg8ez3rweoz4zmnmpv4c&dl=0https:/****), [colab](https://colab.research.google.com/drive/1DiUZdVIhjLCmQfOm2o7qI9eKzaGOxGC9?usp=sharinghttps:/) |
