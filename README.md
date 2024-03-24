# OneDiffGenMetrics


This repository is used for evaluating the quality of generation after compilation acceleration using [onediff](https://github.com/siliconflow/onediff).

### TODO

## SDXL


bash run_sdxl_tests.sh

int8+deepcache

e2e (30 steps) elapsed: 3057.6571576595306 s
Loading model ...
Loading model successfully!
-----------benchmark score ---------------- 
eval paintings       28.51       0.4962 (std)
eval photo           26.91       0.4605
eval concept-art     28.42       0.3953
eval anime           30.50       0.3470
eval Average         **28.58** 


int8

e2e (30 steps) elapsed: 7068.445282936096 s
Loading model ...
Loading model successfully!
-----------benchmark score ---------------- 
eval2 paintings       30.05      0.3897
eval2 photo           28.26      0.4339
eval2 concept-art     30.04      0.3807
eval2 anime           31.79      0.3224
eval2 Average         **30.04**


deepcache

e2e (30 steps) elapsed: 3634.662671804428 s
Loading model ...
Loading model successfully!
-----------benchmark score ---------------- 
eval3 paintings       28.45      0.3816
eval3 photo           27.03      0.3348
eval3 concept-art     28.56      0.3517
eval3 anime           30.49      0.3626
eval3 Average         **28.63** 

compile


e2e (30 steps) elapsed: 9043.378895759583 s
Loading model ...
Loading model successfully!
-----------benchmark score ---------------- 
eval4 paintings       30.07      0.3789
eval4 photo           28.42      0.2491
eval4 concept-art     30.17      0.2834
eval4 anime           31.73      0.3485
eval4 Average         **30.10** 


torch



### References

- Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., & Li, H. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv preprint arXiv:2306.09341.
