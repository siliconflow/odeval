# odeval (WIP...)

<p align="center">
<img src="imgs/onediff_logo.png" height="100">
</p>

1. [Introduction](#introduction) 🌟
2. [Installation](#installation) 🛠️
   - [Prepare the OneDiff Environment](#prepare-the-onediff-environment)
   - [Prepare Benchmark Environment](#prepare-benchmark-environment)
3. [Quick Start](#quick-start) ⚡
   - [Generate Benchmark Images](#generate-benchmark-images)
   - [Testing Using Multiple Indicators](#testing-using-multiple-indicators)
4. [Qualitative Evaluation](#qualitative-evaluation) 🎨
5. [References](#references) 📚
6. [Citing](#citing) 📖

## Introduction

This repository is used for evaluating the quality of generation after compilation acceleration using [OneDiff](https://github.com/siliconflow/onediff).

It can also serve as a benchmark for evaluating the performance of different text-to-image models.


## Installation

1. **Prepare the OneDiff environment.**

    Follow the instructions to install OneDiff and other dependencies:
   - https://github.com/siliconflow/onediff/tree/main?tab=readme-ov-file#installation

2. **Prepare Benchmark environment.**

    ```
    pip3 install -r requirements.txt
    pip3 install -e .
    ```


## Quick Start

Evaluating the use of all generative models is divided into two steps, taking the [kolors](https://huggingface.co/Kwai-Kolors/Kolors) model as an example:

### 1. Generate benchmark images.

   - On [MS COCO-30K](https://huggingface.co/datasets/sayakpaul/coco-30-val-2014):

      Assume that the folders `kolors_torch_coco`, `kolors_oneflow_coco`, and `kolors_nexfort_coco` respectively store the original images, images compiled by the onediff's oneflow backend, and images compiled by the nexfort backend.

      ```
      # Create a path to store the generated images.
      mkdir /path/to/your/kolors_torch_coco
      ```

      ```
      # Original pytorch generates reference images.
      python3 models/kolors/text_to_image_kolors_quality_benchmark.py \
      --dataset coco \
      --csv-file resources/MS-COCO_val2014_30k_captions.csv \
      --output-dir /path/to/your/kolors_torch_coco
      ```

      ```
      # Accelerate using onediff's oneflow backend.
      python3 models/kolors/text_to_image_kolors_quality_benchmark.py \
      --compiler oneflow \
      --dataset coco \
      --csv-file resources/MS-COCO_val2014_30k_captions.csv \
      --output-dir /path/to/your/kolors_oneflow_coco
      ```

      ```
      # Accelerate using onediff's nexfort backend.
      python3 models/kolors/text_to_image_kolors_quality_benchmark.py \
      --compiler nexfort \
      --compiler-config '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}' \
      --dataset coco \
      --csv-file resources/MS-COCO_val2014_30k_captions.csv \
      --output-dir /path/to/your/kolors_nexfort_coco
      ```

   - On [Human Preference Dataset v2 (HPD v2)](https://github.com/siliconflow/odeval/wiki/Datasets-and-evaluation-metrics-used-for-quality-benchmarking):

      Simply modify the `--dataset` parameters, do not read prompts from the `--csv-file` parameter, and customize the `--output-dir` for generating images. For example:

      ```
      python3 models/kolors/text_to_image_kolors_quality_benchmark.py \
      --dataset hps \
      --output-dir /path/to/your/kolors_torch_hps
      ```

### 2. Test using multiple indicators with scripts.


   ```
   bash scripts/run_kolors_tests.sh coco
   bash scripts/run_kolors_tests.sh hps
   ```

A quality report can refer to: [models/kolors/README.md](models/kolors/README.md)

## Qualitative evaluation

We collected several typical prompts to visualize the generated images for qualitative evaluation. These prompts reflect the model's semantic understanding, long text, detail, spatial relationships, diversity, clarity, and text embedding capabilities.

- English: [resources/prompts.txt](resources/prompts.txt)

- Chinese: [resources/prompts_cn.txt](resources/prompts_cn.txt)


## References

- Maximilian Seitzer. Compute FID scores with PyTorch. https://github.com/mseitzer/pytorch-fid. 2020
- Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.
- SUN Zhengwentai. clip-score: CLIP Score for PyTorch. https://github.com/Taited/clip-score, 2023.
- Christoph Schuhmann. CLIP+MLP Aesthetic Score Predictor. https://github.com/christophschuhmann/improved-aesthetic-predictor, 2022.
- Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. NeurIPS, 29, 2016.


## Citing

```
@misc{odeval,
  author = {Xiang Li and others},
  title = {odeval: A Library for benchmarking the accelerated generation quality},
  year = {2023},
  publisher = {SiliconFlow},
  howpublished = {\url{https://github.com/siliconflow/odeval}},
  note = {Accessed: 2024-07-26}
}
```
