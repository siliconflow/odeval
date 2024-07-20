# Using onediff to accelerate the quality evaluation of kolors.


## Qualitative evaluation

We used several typical prompts to visualize generated images, proving that the compilation acceleration of onediff is almost lossless.

| Prompt | Reference Image | OnFlow Backend | NexFort Backend |
|--------|-----------------|----------------|-----------------|
| 穿着黑色T恤的可爱小狗，T恤上面中文绿色大字写着“可图”。 | <img src="./asset/kolors_01.png" width="150px"> | <img src="./asset/kolors_oneflow_compile_01.png" width="150px"> | <img src="./asset/kolors_nexfort_compile_01.png" width="150px"> |
| 一张兔子的特写照片，春天的森林中，有雾、晕影、开花、戏剧性的氛围，以三分法为中心，200mm 1.4f的微距镜头拍摄。 | <img src="./asset/kolors_02.png" width="150px"> | <img src="./asset/kolors_oneflow_compile_02.png" width="150px"> | <img src="./asset/kolors_nexfort_compile_02.png" width="150px"> |
| 一条繁忙的街道，车辆在两个方向上行驶，街道上有几辆双层巴士和周围的人们。 | <img src="./asset/kolors_03.png" width="150px"> | <img src="./asset/kolors_oneflow_compile_03.png" width="150px"> | <img src="./asset/kolors_nexfort_compile_03.png" width="150px"> |
| 动漫风格，一名女孩在室内，坐在客厅的沙发上，拥有粉红色的头发、白色的袜子、蓝色的眼睛，从背后、从上方看，面向观众，正在玩视频游戏，手持控制器，穿着黑色丝绸，嘴唇微张。 | <img src="./asset/kolors_04.png" width="150px"> | <img src="./asset/kolors_oneflow_compile_04.png" width="150px"> | <img src="./asset/kolors_nexfort_compile_04.png" width="150px"> |
| 一张年轻中国女性在公园的肖像照片。她的长黑发轻轻飘动。背景中柔和的樱花粉增添了画面的美感。 | <img src="./asset/kolors_05.png" width="150px"> | <img src="./asset/kolors_oneflow_compile_05.png" width="150px"> | <img src="./asset/kolors_nexfort_compile_05.png" width="150px"> |



## Quantitative evaluation
