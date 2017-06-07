# Model

## Introduction

본 연구는 우주 행성의 모양을 보고 판단하는 것을 목적으로 한다. **그 중 이 문서는 Decision Tree 중 첫번째 트리에 대해서만 Regression 예측을 수행한 결과를 설명하였고, CNN 모델의 정상 동작됨을 확인하는데 그 목적을 두고 있다.** 

- TensorFlow 1.1을 사용하고 개발하였음
- TensorBoard를 통해 각 변수들의 상태를 실시간으로 확인하였음

## Model
- Fully Convolutional Network을 기본으로 하여 모델을 개발하였음
- 초기화를 위해 Keming He의 방식을 사용함
- learning rate은 현재 1e-5로 하고 있음. BN(Batch Normalization)을 하지 않은 경우 1e-6으로 해야 어느 정도(50% 만 정상동작) 동작했지만, BN 적용 후에는 learning rate을 올려도 정상 동작 함 
- Optimazier로는 Adam Optimizer 사용함
- TODO : Model 그림 추가

## Code 

### Overall Architecture

### How to use TensorBoard

```
$ tensorboard --logdir=[logdirectory 위치] --port=[번호,Option,Default=6006]
```


## Test

### Test #1
- 목적: Batch Normalization을 수행한 경우와 안한 경우 테스트

### Test #2
- 목적: Color 영상과 Gray 영상를 입력으로 한 경우 성능 측정

### Test #3
- 목적: Convolution Layer의 갯수를 얼마까지 줄여도 되는지 성능 측정

### Test #4
- 목적: 


## Reference
- [Sander Dieleman (1st)](https://github.com/benanne/kaggle-galaxies)
- [Tund (3rd)](https://github.com/tund/kaggle-galaxy-zoo)

