# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: 623
./splitted_dataset/train: 498
./splitted_dataset/train/<OK>: 262​
./splitted_dataset/train/<NO>: 236​
./splitted_dataset/val: 125
./splitted_dataset/train/<OK>: 65
./splitted_dataset/train/<NO>: 60​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S|1.0000|54.35|0:00:51.961996|32|0.0071|--|
|EfficientNet-B0|1.0000|167.85|0:00:26.001007|32|0.0049|--| 
|DeiT-Tiny|1.0000|52.98|0:00:22.579750|32|0.0001|--|
|MobileNet-V3-large-1x|1.000|224.82|0:00:14.860949|32|0.0058|--|


## FPS 측정 방법
frame = 현재시간 - 이전시간
fps = 1/frame
