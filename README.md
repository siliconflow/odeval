# odeval (WIP...)

<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>


This repository is used for evaluating the quality of generation after compilation acceleration using [OneDiff](https://github.com/siliconflow/onediff).

1. [Quick Start](#quick-start)
   - [Set up the OneDiff environment](#1-set-up-the-onediff-environment)
   - [Prepare Benchmark environment](#2-prepare-benchmark-environment)
2. [Models](#models)
   - [Introduction](#introduction)
   - [SDXL](#sdxl)
   - [SD 1.5](#sd-15)
   - [SVD](#svd)
4. [References](#references)


## Quick Start

1. **Prepare the OneDiff environment.**

    Follow the instructions to install OneDiff and other dependencies: 
- [Community Edition (CE)](https://github.com/siliconflow/onediff/tree/main?tab=readme-ov-file#installation)
- [Enterprise Edition (EE)](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#diffusers-with-onediff-enterprise)

2. **Prepare Benchmark environment.**

    The dataset used for quality benchmarking is the [Human Preference Dataset v2 (HPD v2)](https://huggingface.co/datasets/zhwang/HPDv2), which will be automatically downloaded when any script is executed.

    <p align="center"><img src="imgs/overview.png"/ width="70%"><br></p>

    Install the benchmark library:
    ```
    pip3 install -r requirements.txt
    pip3 install -e .
    ```




## Models

### Introduction
Currently, the quality of SDXL, SD 1.5, and SVD after OneDiff acceleration has been benchmarked. For explanations of these metrics, please see: https://github.com/siliconflow/odeval/wiki/Datasets-and-evaluation-metrics-used-for-quality-benchmarking.

### SDXL
Run:

    ```
    bash run_sdxl_tests.sh
    ```

**HPS v2** comparison results:


| Optimization Technique | Paintings  | Photo | Concept-Art  | Anime | Average Score |
|------------------------|-----------------|-------------|-------------------|-------------|---------------|
| OneDiff Quant + OneDiff DeepCache (EE) | 26.58 ± 0.4468  | 24.31 ± 0.4500 | 26.55 ± 0.2888  | 28.81 ± 0.3119 | 26.56 | 
| OneDiff DeepCache (CE) | 26.61 ± 0.4333  | 24.34 ± 0.4189 | 26.61 ± 0.2270  | 28.84 ± 0.3113 | 26.60 | 
| OneDiff Quant (EE)  | 27.87 ± 0.4419  | 25.70 ± 0.4253 | 27.86 ± 0.2222  | 29.93 ± 0.3920 | 27.84 | 
| OneDiff Compile (CE) | 27.84 ± 0.4312  | 25.70 ± 0.4550 | 27.87 ± 0.2638  | 29.91 ± 0.3791 | 27.83 | 
| Pytorch | 27.82 ± 0.4275  | 25.70 ± 0.4534 | 27.85 ± 0.2432  | 29.92 ± 0.3666 | 27.82 | 

> [!NOTE]
Scores for four styles ("Animation", "Concept-art", "Painting", and "Photo") and the average score are provided. Higher scores indicate better image quality.

| Optimization Technique | SSIM   | MSE    | MAE    |
|--------|--------|--------|--------|
| OneDiff Quant + OneDiff DeepCache (EE) | 0.7483 | 76.123 | 93.163 |
| OneDiff DeepCache (CE)  | 0.7504 | 76.198 | 92.085 |
| OneDiff Quant (EE) | 0.8794 | 30.664 | 117.736|
| OneDiff Compile (CE) | 0.9380 | 16.155 | 93.989 |
| Pytorch | -  | - | -  | - | - | 

<details>
<summary>CLIP Score comparison results:</summary>

   | Optimization Technique | Paintings | Photo | Concept-Art| Anime | Average Score |
   |--------------------------------------|-----------------|-------------|-------------------|-------------|---------------|
   | OneDiff Quant + OneDiff DeepCache (EE) | 35.46 | 34.44 | 35.24 | 31.85 | 34.25 |
   | OneDiff DeepCache (CE) | 35.42 | 34.47 | 35.15 | 31.83 | 34.22 |
   | OneDiff Quant (EE) | 35.88 | 34.74 | 35.53 | 31.80 | 34.49 |
   | OneDiff Compile (CE) | 35.78 | 34.83 | 35.43 | 31.77 | 34.45 |
   | Pytorch | 35.78 | 34.83 | 35.42 | 31.77 | 34.45 |
   </details>

<details>
<summary>Average Aesthetic Score and Inception Score comparison results:</summary>

   | Optimization Technique | Average Aesthetic Score | Average Inception Score |
   |---------------------------------------|-------------------------|------------------------------|
   | OneDiff Quant + OneDiff DeepCache (EE) | 5.93                    | 16.43 ± 3.75                 |
   | OneDiff DeepCache (CE)                | 5.91                    | 15.82 ± 3.80                 |
   | OneDiff Quant (EE)                    | 5.97                    | 16.02 ± 4.60                 |
   | OneDiff Compile (CE)                  | 5.97                    | 15.88 ± 4.43                 |
   | Pytorch                               | 5.97                    | 15.80 ± 4.24                 |
   </details>


### SD 1.5
Run:

    ```
    bash run_sd1_5_tests.sh
    ```

**HPS v2** comparison results:

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
|-------------------------|-----------------|-------------|-------------------|-------------|---------------|
| OneDiff Quant + OneDiff DeepCache (EE) | 24.11 ± 0.2549  | 25.10 ± 0.5905 | 24.07 ± 0.3959  | 25.45 ± 0.2102 | 24.68         |  
| OneDiff DeepCache (CE)               | 23.88 ± 0.4237  | 25.23 ± 0.4587 | 23.96 ± 0.4445  | 25.51 ± 0.2846 | 24.65         | 
| OneDiff Quant (EE)           | 24.68 ± 0.2271  | 25.54 ± 0.5553 | 24.73 ± 0.3563  | 26.02 ± 0.4202 | 25.24         |
| OneDiff Compile (CE)         | 24.58 ± 0.3372  | 25.83 ± 0.3850 | 24.71 ± 0.4705  | 26.25 ± 0.2840 | 25.34         | 
| Pytorch                      | 24.55 ± 0.3336  | 25.78 ± 0.3986 | 24.70 ± 0.4624  | 26.24 ± 0.2989 | 25.32         | 

<details>
<summary>CLIP Score comparison results:</summary>

   | Optimization Technique               | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
   |--------------------------------------|-----------------|-------------|-------------------|-------------|---------------|
   | OneDiff Quant + OneDiff DeepCache (EE)       | 33.55           | 32.72       | 33.57             | 30.87       | 32.68         |
   | OneDiff DeepCache (CE)                       | 33.62           | 32.79       | 33.48             | 30.96       | 32.71         |
   | OneDiff Quant (EE)                   | 33.64           | 32.84       | 33.72             | 30.79       | 32.75         |
   | OneDiff Compile (CE)                 | 33.75           | 33.00       | 33.63             | 30.91       | 32.82         |
   | Pytorch                              | 33.76           | 32.98       | 33.62             | 30.96       | 32.83         |
   </details>


<details>
<summary>Average Aesthetic Score and Inception Score comparison results:</summary>

   | Optimization Technique                | Average Aesthetic Score | Average Inception Score      |
   |---------------------------------------|-------------------------|------------------------------|
   | OneDiff Quant + OneDiff DeepCache (EE) | 5.43                    | 14.71 ± 3.70                 |
   | OneDiff DeepCache (CE)                | 5.42                    | 15.30 ± 4.59                 |
   | OneDiff Quant (EE)                    | 5.46                    | 15.05 ± 4.31                 |
   | OneDiff Compile (CE)                  | 5.46                    | 15.20 ± 4.07                 |
   | Pytorch                               | 5.46                    | 15.25 ± 4.49                 |
   </details>




### SVD

Run:

    ```
    bash run_svd_tests.sh
    ```

> [!NOTE]
Evaluate using the last frame of the video.

**HPS v2** comparison results:

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
|-----------------------------|-----------------|-------------|-------------------|-------------|---------------|
| OneDiff DeepCache (CE)               | 24.72 ± 0.1604 | 22.77 ± 0.0308 | 25.15 ± 0.2523 | 25.00 ± 1.0273 | 24.41         | 
| OneDiff Quant + OneDiff DeepCache (EE) | 24.72 ± 0.0327 | 22.81 ± 0.0881 | 25.25 ± 0.0405 | 25.19 ± 0.8912 | 24.49        | 
| OneDiff Compile (CE)         | 25.84 ± 0.0566 | 24.54 ± 0.1882 | 26.43 ± 0.0194 | 26.79 ± 0.5265 | 25.90        | 
| Pytorch                      | 25.82 ± 0.1076 | 24.28 ± 0.1298 | 26.48 ± 0.0792 | 26.82 ± 0.5806 | 25.85         | 


<details>
<summary>CLIP Score comparison results:</summary>

   | Optimization Technique           | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
   |----------------------------------|-----------------|-------------|-------------------|-------------|---------------|
   | OneDiff DeepCache (CE)                   | 31.75           | 30.52       | 30.68             | 29.42       | 30.59         |
   | OneDiff Quant + OneDiff DeepCache (EE)   | 31.82           | 30.54       | 30.83             | 29.38       | 30.64         |
   | OneDiff Compile (CE)             | 32.57           | 31.38       | 31.66             | 30.02       | 31.41         |
   | Pytorch                          | 32.43           | 31.24       | 31.81             | 29.92       | 31.35         |
   </details>

<details>
<summary>Average Aesthetic Score and Inception Score comparison results:</summary>

   | Optimization Technique                | Average Aesthetic Score | Average Inception Score   |
   |---------------------------------------|-------------------------|---------------------------|
   | OneDiff DeepCache (CE)                | 5.32                    | 7.63 ± 2.19               |
   | OneDiff Quant + OneDiff DeepCache (EE) | 5.31                    | 7.86 ± 2.25               |
   | OneDiff Compile (CE)                  | 5.48                    | 8.18 ± 2.33               |
   | Pytorch                               | 5.50                    | 7.88 ± 1.97               |
   </details>



### References

- Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.
- SUN Zhengwentai. clip-score: CLIP Score for PyTorch. https://github.com/Taited/clip-score, 2023.
- Christoph Schuhmann. CLIP+MLP Aesthetic Score Predictor. https://github.com/christophschuhmann/improved-aesthetic-predictor, 2022.
- Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. NeurIPS, 29, 2016.
