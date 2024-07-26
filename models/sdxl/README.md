# WIP...

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
