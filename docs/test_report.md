# Test Report
## Until now
### purpose & background
- deep learning으로 galaxy classification result에 대한 regression
- kaggle contest : input galaxy에 대하여 총 11가지 질문과 그에 따른 37가지의 대답들로 구성되어 있는 decision tres의 각 node의 probability를 예측
- 각각의 answer는 question에 따른 확률 값을 지님
  (ex.A11+A12+A13 = 1,  A91+A92+A93=A21, A51+...+A54=A22 ...  )
- 기존에 은하의 type에 따른 분류와는 차이가 있음 -> input image가 어떤 type의 은하인지 구별하는것이 아니고, citizen이 유추한 각 트리 단계의 값을 얻는것(regression 문제)

### Data 
- SDSS galaxy image 424 x 424 (training set : 61578, test set : )


### Test environment
- 4 GPU & 12 GB memory
- model : 7 conv layers + 2 fully connected layers
- 

### Method
- pre-processing 
   -  중앙을 기준으로 image를 207x207로 crop -> 128 x 128 down sampling
   -  rgb color 이용
   -  

## Furture works
1. data augmentation 
   - radial 방향의 intensity 분포를 중심으로 급격하게 변하는 지점을 따라 image crop -> background와 galaxy 부분 구분
   - image rotation
   - image translation, reflection
   - add to noise  
2. Refer to other study 
   - 2016 Tuccilo : ellipticity (catalog vs model) fitting 
   - 2014 Chou : extract = PCA, SIFT.., regression = standard
    regression method, ridge regression method,
   - **2016 Edwrad J Kim: leak ReLU, VGGNet, Tree for probabilistic Classification**
3. Constraint 
   - 질문에 따른 값의 제한을 둠으로써 (A91+A92+A93=A21) 초기값 및 결과값의 한계를 제공
4. classification -> detection
5. 
