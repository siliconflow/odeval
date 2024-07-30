# Using onediff to accelerate the quality evaluation of SDXL-Lightning.

### Prompt
"product photography, world of warcraft orc warrior, white background"

### Results
|               | Image |
|------------------------|-------|
| **PyTorch**            | <img src="./asset/sdxl_light.png" width="300px"> |
| **OneFlow Compile**    | <img src="./asset/sdxl_light_oneflow_compile.png" width="300px"> |
| **OneFlow Quantization**| <img src="./asset/sdxl_light_oneflow_quant.png" width="300px"> |
| **NexFort Compile**    | <img src="./asset/sdxl_light_nexfort_compile.png" width="300px"> |
| **NexFort Quantization**| <img src="./asset/sdxl_light_nexfort_quant.png" width="300px"> |
