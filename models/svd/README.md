# WIP...

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
