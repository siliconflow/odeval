# WIP...

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
