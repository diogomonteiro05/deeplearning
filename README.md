# 🖼️ DeepFake Detection & Generation – Deep Learning Project

## 📖 Overview
This project aims to develop deep learning models for both **deepfake detection** and **deepfake generation** using the **DeepFakeFace (DFF) dataset**. The dataset consists of **30,000 real images** from the IMDB-WIKI dataset and **90,000 fake images** generated by **Stable Diffusion v1.5, Stable Diffusion Inpainting, and InsightFace models**. The project explores **discriminative** and **generative** deep learning approaches to classify and synthesize deepfake images.

## 📌 Project Scope & Methodology
✅ **Deepfake Classification**: Train a discriminative model (CNN, Vision Transformer, or MLP) to classify real vs. fake images.  
✅ **Deepfake Generation**: Develop a generative model (GAN, VAE, or Diffusion Model) to create realistic fake images based on the DeepFakeFace dataset.  
✅ **Dataset Handling**: Utilize the DeepFakeFace dataset for training and evaluation, with real images sourced from the IMDB-WIKI dataset and fake images generated by **Stable Diffusion**, **Inpainting**, and **InsightFace**.  
✅ **Performance Evaluation**: Assess classifier performance with accuracy, precision, recall, and F1-score. Evaluate generative model quality using **FID (Fréchet Inception Distance)** and **Inception Score** to measure the quality of generated images.  
✅ **Iterative Model Improvement**: Implement data augmentation, hyperparameter tuning, and regularization techniques for performance enhancement.  
✅ **Advanced Techniques (Bonus)**: Investigate **self-supervised learning, uncertainty quantification, and federated learning** for potential improvements.  


## Setting up the data
The dataset is available in the following link: https://huggingface.co/datasets/OpenRL/DeepFakeFace.



<!--TODO:
The code in this project assumes that the data is inside the folder `data`, organized in the following way:
-->
