# Galaxies Classification

## 1. Reference
- Galaxyzoo(http://www.zooniverse.org/)의 data 및 구조를 따름
(※참조 논문 (Willett et al. (2013),https://arxiv.org/abs/1308.3496v2) )

- Kaggle의 galaxyzoo 대회의 data set을 활용
(https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)

## 2. Idea
- 기존의 kaggle 대회는 decision tree의 각 probability를 유추 하는것으로 진행되었음
- 차별화(TBD)
 - rotation invariant 향상 
 - classification -> object detection
 - data quality 향상 위해 intensity level을 radial 방향으로 확인하여 대상 은하만을 중심으로 data 추출하여 재구성
   (문제점 : 중심 은하 size가 각기 달라서 data 마다 크기가 달라짐)
