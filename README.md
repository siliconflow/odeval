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

    ```
    # Method 1: Pypi download and install
    pip install hpsv2

    # Method 2: install locally
    git clone https://github.com/tgxs002/HPSv2.git
    cd HPSv2
    pip install -e . 

    # Optional: images for reproducing our benchmark will be downloaded here
    # default: ~/.cache/hpsv2/
    export HPS_ROOT=/your/cache/path
    ```
[HPSv2](https://github.com/tgxs002/HPSv2) is a scoring model that can more accurately predict human preferences on text-generated images.

<p align="center"><img src="imgs/overview.png"/ width="70%"><br></p>

## Models

### SDXL
Run:

    ```
    bash run_sdxl_tests.sh
    ```

HPSv2 comparison results:


| Optimization Technique | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 1024*1024 (h:min:s) |
|------------------------|-----------------|-------------|-------------------|-------------|---------------|--------------------------------|
| OneDiff Quant + DeepCache (EE)     | 28.51 ± 0.4962  | 26.91 ± 0.4605 | 28.42 ± 0.3953  | 30.50 ± 0.3470 | 28.58         | 0:50:57                        |
| OneDiff Quant (EE)                    | 30.05 ± 0.3897  | 28.26 ± 0.4339 | 30.04 ± 0.3807  | 31.79 ± 0.3224 | 30.04         | 1:57:48                     |
| DeepCache (CE)              | 28.45 ± 0.3816  | 27.03 ± 0.3348 | 28.56 ± 0.3517  | 30.49 ± 0.3626 | 28.63         | 1:0:34                    |
| OneDiff Compile (CE)                | 30.07 ± 0.3789  | 28.42 ± 0.2491 | 30.17 ± 0.2834  | 31.73 ± 0.3485 | 30.10         | 2:30:43                       |
| Pytorch                  | 30.07 ± 0.3887  | 28.43 ± 0.2726 | 30.16 ± 0.2686  | 31.74 ± 0.3691 | 30.10         | 3:42:15                      |

> [!NOTE]
Scores for four styles ("Animation", "Concept-art", "Painting", and "Photo") and the average score are provided. Higher scores indicate better image quality.

### SD 1.5
Run:

    ```
    bash run_sd1_5_tests.sh
    ```

HPSv2 comparison results:

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 512*512 (h:min:s) |
|-------------------------|-----------------|-------------|-------------------|-------------|---------------|-------------------------|
| OneDiff Quant + DeepCache (EE) | 24.11 ± 0.2549  | 25.10 ± 0.5905 | 24.07 ± 0.3959  | 25.45 ± 0.2102 | 24.68         | 0:14:01                |
| OneDiff Quant (EE)           | 24.68 ± 0.2271  | 25.54 ± 0.5553 | 24.73 ± 0.3563  | 26.02 ± 0.4202 | 25.24         | 0:23:51                |
| DeepCache (CE)               | 23.88 ± 0.4237  | 25.23 ± 0.4587 | 23.96 ± 0.4445  | 25.51 ± 0.2846 | 24.65         | 0:15:01               |
| OneDiff Compile (CE)         | 24.58 ± 0.3372  | 25.83 ± 0.3850 | 24.71 ± 0.4705  | 26.25 ± 0.2840 | 25.34         | 0:27:27                |
| Pytorch                      | 24.55 ± 0.3336  | 25.78 ± 0.3986 | 24.70 ± 0.4624  | 26.24 ± 0.2989 | 25.32         | 0:51:25                |



### References

- Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.
