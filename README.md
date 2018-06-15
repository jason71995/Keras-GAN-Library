# Keras GAN library

## Introduction
Implementation of GAN papers, all using cifar10 dataset in this project.

ATTENTION:
To compare the differences of GAN methods, the hyperparameters in this project are not exactly same as papers.
Architecture of generators and discriminators are as similar as possible, and using same optimizer setting.

## Environment

```
python==3.6
tensorflow==1.4.0
keras==2.1.0
```

##  Implemented Papers

 - DCGAN - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks [link](https://arxiv.org/abs/1511.06434)
 - LSGAN - Loss-Sensitive Generative Adversarial Networks on Lipschitz Densities [link](https://arxiv.org/abs/1701.06264)
 - WGAN-GP - Improved Training of Wasserstein GANs [link](https://arxiv.org/abs/1704.00028)
 - SNGAN - Spectral Normalization for Generative Adversarial Networks [link](https://arxiv.org/abs/1802.05957)
 
## Results

| Name | 50 epochs |
| :---: | :---: |
| DCGAN | ![alt text](https://i.imgur.com/e3xxYkw.png "DCGAN") |
| LSGAN | ![alt text](https://i.imgur.com/25fLTTR.png "LSGAN") |
| WGAN-GP | ![alt text](https://i.imgur.com/Az9l0FS.png "WGAN-GP") |
| SNGAN | ![alt text](https://i.imgur.com/xfIDVTo.png "SNGAN") |