# RA-CCGAN

## Abstract:
Deep anomaly detection is increasingly popular but requires significant resources. Recent research has focused on efficient GANs for anomaly detection. However, the emphasis on generating realistic samples over integrating conditional information limits their ability to detect unique anomaly features.
In this study, we propose a unified optimized and efficient framework for unsupervised anomaly detection. We introduce a Residual Attention Class-Conditional GAN (RA-CCGAN) architecture, which is composed of two modules, one is based on class conditional generative adversarial net for data generation and an encoder is for fast anomaly detection training. Specifically, the generator can produce target-style realistic images that correspond accurately with designated pathology regions. The discriminator can discriminate between real and fake samples of the specified class condition or pathology. And the encoder network is utilized to facilitate a fast mapping from images space to the class's latent space of specific class condition for evaluating unseen images. Moreover, we replace the standard convolutions in our model by Depthwise Separble Convolution (DSC) to reduce computational and hardware resources. Indeed, we incorporate the Convolutional Block Attention Module (CBAM) into our model to enable a focus on pertinent regions of the image without increasing the parameters or computational complexity. We take pulmonary anomalies detection as our task and conduct experiments on two datasets. Finally, the effectiveness of the approach is demonstrated on three benchmarking datasets, namely Digit MNIST, Fashion MNIST and MVTec. The achieved results at the image level showcase state-of-the-art performance. 


## Proposed Framework
![Full_architecture](https://github.com/nnajeh/RA-CCGAN/assets/38373885/d5f2566d-b3bc-45c0-a72b-4f2744f41501)

This is a PyTorch/GPU implementation of the paper [Unifying Class-Conditional GAN Framework for Unsupervised Pulmonary Anomalies Detection]
