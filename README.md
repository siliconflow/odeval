# OneDiffGenMetrics

<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

This repository is used for evaluating the quality of generation after compilation acceleration using [OneDiff](https://github.com/siliconflow/onediff).

## Quick Start

1. **Prepare the OneDiff environment.**

    Follow the instructions to install OneDiff and other dependencies. 
- [Community Edition (CE)](https://github.com/siliconflow/onediff/tree/main?tab=readme-ov-file#installation)
- [Enterprise Edition (EE)](https://github.com/siliconflow/onediff/blob/main/README_ENTERPRISE.md#diffusers-with-onediff-enterprise)

2. **Prepare Benchmark environment.**

    Human Preference Dataset v2 (HPD v2)
    [HPSv2](https://github.com/tgxs002/HPSv2) is a scoring model that can more accurately predict human preferences on text-generated images.

<p align="center"><img src="imgs/overview.png"/ width="70%"><br></p>

    ```
    pip3 install -r requirements.txt
    ```




## Models

### SDXL
Run:

    ```
    bash run_sdxl_tests.sh
    ```

**HPSv2 comparison results:**


| Optimization Technique | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 1024*1024 (h:min:s) |
|------------------------|-----------------|-------------|-------------------|-------------|---------------|--------------------------------|
| OneDiff Quant + DeepCache (EE)     | 28.51 ± 0.4962  | 26.91 ± 0.4605 | 28.42 ± 0.3953  | 30.50 ± 0.3470 | 28.58         | 0:50:57                        |
| OneDiff Quant (EE)                    | 30.05 ± 0.3897  | 28.26 ± 0.4339 | 30.04 ± 0.3807  | 31.79 ± 0.3224 | 30.04         | 1:57:48                     |
| DeepCache (CE)              | 28.45 ± 0.3816  | 27.03 ± 0.3348 | 28.56 ± 0.3517  | 30.49 ± 0.3626 | 28.63         | 1:0:34                    |
| OneDiff Compile (CE)                | 30.07 ± 0.3789  | 28.42 ± 0.2491 | 30.17 ± 0.2834  | 31.73 ± 0.3485 | 30.10         | 2:30:43                       |
| Pytorch                  | 30.07 ± 0.3887  | 28.43 ± 0.2726 | 30.16 ± 0.2686  | 31.74 ± 0.3691 | 30.10         | 3:42:15                      |


> [!NOTE]
Scores for four styles ("Animation", "Concept-art", "Painting", and "Photo") and the average score are provided. Higher scores indicate better image quality.
Inference Time testing is conducted across the entire benchmark dataset.


**Calculating CLIP Score:**

| Optimization Technique               | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
|--------------------------------------|-----------------|-------------|-------------------|-------------|---------------|
| OneDiff Quant + DeepCache (EE)       | 35.46           | 34.44       | 35.24             | 31.85       | 34.25         |
| OneDiff Quant (EE)                   | 35.88           | 34.74       | 35.53             | 31.80       | 34.49         |
| DeepCache (CE)                       | 35.42           | 34.47       | 35.15             | 31.83       | 34.22         |
| OneDiff Compile (CE)                 | 35.78           | 34.83       | 35.43             | 31.77       | 34.45         |
| Pytorch                              | 35.78           | 34.83       | 35.42             | 31.77       | 34.45         |

**Average Aesthetic Score:**

| Optimization Technique  | Average Aesthetic Score |
|-----|-------------------------|
| OneDiff Quant + DeepCache (EE)   | 5.93                    |
| OneDiff Quant (EE)   | 5.97                    |
| DeepCache (CE)   | 5.91                    |
| OneDiff Compile (CE)   | 5.97                    |
| Pytorch   | 5.97                    |

**Average Inception Score:**

| Optimization Technique  | Average Inception Score |
|-------|------------------|
| OneDiff Quant + DeepCache (EE)     | 16.43 ± 3.75     |
| OneDiff Quant (EE)     | 16.02 ± 4.60     |
| DeepCache (CE)     | 15.82 ± 3.80     |
| OneDiff Compile (CE)     | 15.88 ± 4.43     |
| Pytorch     | 15.80 ± 4.24     |



### SD 1.5
Run:

    ```
    bash run_sd1_5_tests.sh
    ```

**HPSv2 comparison results:**

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 512*512 (h:min:s) |
|-------------------------|-----------------|-------------|-------------------|-------------|---------------|-------------------------|
| OneDiff Quant + DeepCache (EE) | 24.11 ± 0.2549  | 25.10 ± 0.5905 | 24.07 ± 0.3959  | 25.45 ± 0.2102 | 24.68         | 0:14:01                |
| OneDiff Quant (EE)           | 24.68 ± 0.2271  | 25.54 ± 0.5553 | 24.73 ± 0.3563  | 26.02 ± 0.4202 | 25.24         | 0:23:51                |
| DeepCache (CE)               | 23.88 ± 0.4237  | 25.23 ± 0.4587 | 23.96 ± 0.4445  | 25.51 ± 0.2846 | 24.65         | 0:15:01               |
| OneDiff Compile (CE)         | 24.58 ± 0.3372  | 25.83 ± 0.3850 | 24.71 ± 0.4705  | 26.25 ± 0.2840 | 25.34         | 0:27:27                |
| Pytorch                      | 24.55 ± 0.3336  | 25.78 ± 0.3986 | 24.70 ± 0.4624  | 26.24 ± 0.2989 | 25.32         | 0:51:25                |

**Calculating CLIP Score:**

| Optimization Technique               | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
|--------------------------------------|-----------------|-------------|-------------------|-------------|---------------|
| OneDiff Quant + DeepCache (EE)       | 33.55           | 32.72       | 33.57             | 30.87       | 32.68         |
| OneDiff Quant (EE)                   | 33.64           | 32.84       | 33.72             | 30.79       | 32.75         |
| DeepCache (CE)                       | 33.62           | 32.79       | 33.48             | 30.96       | 32.71         |
| OneDiff Compile (CE)                 | 33.75           | 33.00       | 33.63             | 30.91       | 32.82         |
| Pytorch                              | 33.76           | 32.98       | 33.62             | 30.96       | 32.83         |


**Average Aesthetic Score:**

| Optimization Technique              | Average Aesthetic Score |
|-------------------------------------|-------------------------|
| OneDiff Quant + DeepCache (EE)      | 5.43                    |
| OneDiff Quant (EE)                  | 5.46                    |
| DeepCache (CE)                      | 5.42                    |
| OneDiff Compile (CE)                | 5.46                    |
| Pytorch                             | 5.46                    |

**Average Inception Score:**

| Optimization Technique              | Average Inception Score |
|-------------------------------------|-------------------------|
| OneDiff Quant + DeepCache (EE)      | 14.71 ± 3.70            |
| OneDiff Quant (EE)                  | 15.05 ± 4.31            |
| DeepCache (CE)                      | 15.30 ± 4.59            |
| OneDiff Compile (CE)                | 15.20 ± 4.07            |
| Pytorch                             | 15.25 ± 4.49            |




### SVD

Run:

    ```
    bash run_svd_tests.sh
    ```

**HPSv2 comparison results:**

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time (h:min:s) |
|-----------------------------|-----------------|-------------|-------------------|-------------|---------------|-------------------------|
| OneDiff Quant + DeepCache (EE) | 24.72 ± 0.0327 | 22.81 ± 0.0881 | 25.25 ± 0.0405 | 25.19 ± 0.8912 | 24.49        | 7:28:37                |
| DeepCache (CE)               | 24.72 ± 0.1604 | 22.77 ± 0.0308 | 25.15 ± 0.2523 | 25.00 ± 1.0273 | 24.41         | 7:08:59                |
| OneDiff Compile (CE)         | 25.84 ± 0.0566 | 24.54 ± 0.1882 | 26.43 ± 0.0194 | 26.79 ± 0.5265 | 25.90        | 7:53:22                |
| Pytorch                      | 25.82 ± 0.1076 | 24.28 ± 0.1298 | 26.48 ± 0.0792 | 26.82 ± 0.5806 | 25.85         | 9:53:29                |


**Calculating CLIP Score:**

| Optimization Technique           | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score |
|----------------------------------|-----------------|-------------|-------------------|-------------|---------------|
| OneDiff Quant + DeepCache (EE)   | 31.82           | 30.54       | 30.83             | 29.38       | 30.64         |
| DeepCache (CE)                   | 31.75           | 30.52       | 30.68             | 29.42       | 30.59         |
| OneDiff Compile (CE)             | 32.57           | 31.38       | 31.66             | 30.02       | 31.41         |
| Pytorch                          | 32.43           | 31.24       | 31.81             | 29.92       | 31.35         |


**Average Aesthetic Score:**

| Optimization Technique      | Average Aesthetic Score |
|-----------------------------|-------------------------|
| OneDiff Quant + DeepCache (EE)                 | 5.31                    |
| DeepCache (CE)                  | 5.32                    |
| OneDiff Compile (CE)                | 5.48                    |
| Pytorch                 | 5.50                    |

**Average Inception Score:**

| Optimization Technique      | Average Inception Score    |
|-----------------------------|----------------------------|
| OneDiff Quant + DeepCache (EE)                 | 7.86 ± 2.25                |
| DeepCache (CE)               | 7.63 ± 2.19                |
| OneDiff Compile (CE)                  | 8.18 ± 2.33                |
| Pytorch                 | 7.88 ± 1.97                |



### References

- Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.
