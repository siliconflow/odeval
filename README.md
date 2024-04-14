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


Calculating CLIP Score:
| Optimization Technique | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 1024*1024 (h:min:s) |
|------------------------|-----------------|-------------|-------------------|-------------|---------------|--------------------------------|
| OneDiff Quant + DeepCache (EE)     | 35.45634841918945  | 34.442588806152344 | 35.23531723022461 | 31.84648895263672 | 28.58         | 0:50:57                        |
| OneDiff Quant (EE)                    | 35.8808479309082  | 34.73968505859375 | 35.52915954589844  | 31.801517486572266 | 30.04         | 1:57:48                     |
| DeepCache (CE)              | 35.416053771972656  | 34.47349166870117 | 35.14967727661133  | 31.832393646240234 | 28.63         | 1:0:34                    |
| OneDiff Compile (CE)                | 35.775978088378906  | 34.8296623229980 | 35.43376922607422 | 31.768224716186523 | 30.10         | 2:30:43                       |
| Pytorch                  | 35.7795562744140  | 34.8265266418457 | 35.424835205078125  | 31.77286720275879 | 30.10         | 3:42:15                      |



Average aesthetic score for all images: 5.928362923562527
Average aesthetic score for all images: 5.974049439430237
Average aesthetic score for all images: 5.91144324779510
Average aesthetic score for all images: 5.969082910120488
Average aesthetic score for all images: 5.971234182715416


(16.432745481557347, 3.7464321723009113)
(16.02010587107501, 4.604344672464527)
(15.819354797830417, 3.802340177497511)
(15.87585141502027, 4.429504298365484)
(15.795889694044481, 4.240929938436406)



> [!NOTE]
Scores for four styles ("Animation", "Concept-art", "Painting", and "Photo") and the average score are provided. Higher scores indicate better image quality.
Inference Time testing is conducted across the entire benchmark dataset.

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

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 512*512 (h:min:s) |
|-------------------------|-----------------|-------------|-------------------|-------------|---------------|-------------------------|
| OneDiff Quant + DeepCache (EE) | 33.549373626708984  | 32.72171401977539 | 33.56893539428711  | 30.86512565612793 | 24.68         | 0:14:01                |
| OneDiff Quant (EE)           | 33.64154052734375  | 32.839080810546875 |  33.72417068481445 | 30.787025451660156 | 25.24         | 0:23:51                |
| DeepCache (CE)               | 33.61925506591797 | 32.78925323486328| 33.47829818725586 | 30.957332611083984 | 24.65         | 0:15:01               |
| OneDiff Compile (CE)         | 33.7487869262695  | 33.00041580200195 | 33.62976837158203  | 30.908231735229492 | 25.34         | 0:27:27                |
| Pytorch                      | 33.764949798583984  | 32.97639465332031 | 33.62176513671875 |30.956588745117188 | 25.32         | 0:51:25                |

r all images: 5.427187633961439
ore for all images: 5.458056560158729
Average aesthetic score for all images: 5.423031551316381
Average aesthetic score for all images: 5.464233786761761
Average aesthetic score for all images: 5.464687652438879

(14.708145360622566, 3.7015709571751056)
(15.047029879537984, 4.310061138159499)
(15.302314761365043, 4.591594308119612)
(15.203076023748048, 4.074212125431419)
(15.253643227273168, 4.488194564065999)



### SVD

Run:

    ```
    bash run_svd_tests.sh
    ```

HPSv2 comparison results:

| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 512*512 (h:min:s) |
|-------------------------|-----------------|-------------|-------------------|-------------|---------------|-------------------------|
| OneDiff Quant + DeepCache (EE) | 24.72 ± 0.0327 | 22.81 ± 0.0881 | 25.25 ± 0.0405 | 25.19 ± 0.8912 | 24.62        | 26917.555s                |
| DeepCache (CE)           | 24.72 ± 0.1604 | 22.77 ± 0.0308 | 25.15 ± 0.2523 | 25.00 ± 1.0273 | 24.62         | 25739.851s                |
| OneDiff Compile (CE)         | 25.84 ± 0.0566 | 24.54 ± 0.1882 | 26.43 ± 0.0194 | 26.79 ± 0.5265 | 25.96        | 28402.692s                |
| Pytorch                      | 25.82 ± 0.1076 | 24.28 ± 0.1298 | 26.48 ± 0.0792 | 26.82 ± 0.5806 | 25.96         | 35609.394s                |



| Optimization Technique      | Paintings Score | Photo Score | Concept-Art Score | Anime Score | Average Score | Inference Time for 30 Steps, 512*512 (h:min:s) |
|-------------------------|-----------------|-------------|-------------------|-------------|---------------|-------------------------|
| OneDiff Quant + DeepCache (EE) | 31.822914123535156  | 30.53727912902832 | 30.825557708740234 | 29.38327980041504 | 24.68         | 0:14:01                |
| DeepCache (CE)           | 31.75057601928711  |30.51968765258789 | 30.68446159362793 | 29.423095703125 | 25.24         | 0:23:51                |
| OneDiff Compile (CE)         | 32.57094955444336  | 31.377920150756836 | 31.65696144104004  | 30.022836685180664 | 25.34         | 0:27:27                |
| Pytorch                      | 32.426265716552734  | 31.23940658569336 |  31.80515480041504 | 29.919218063354492 | 25.32         | 0:51:25                |

python -m clip_score /home/lixiang/OneDiffGenMetrics/test/anime prompts/anime


Average aesthetic score for all images: 5.3130029165744785
verage aesthetic score for all images: 5.324187580347061
Average aesthetic score for all images: 5.476414643526077
Average aesthetic score for all images: 5.495331493616104


(7.862378483094004, 2.2547635068701037)
(7.633055066268389, 2.1945581418387143)
(8.18279247932499, 2.328945827026442)
(7.878105649778755, 1.9717273821302463)


### References

- Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.
