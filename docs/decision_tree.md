
# Decision Tree & Data 

![decision tree](./decision_tree.png)

## 1. Decision Tree
galaxy를 분류하기 위한 의사 결정과정으로 zooniverse의 일반인을 대상으로 하여 작성.총 11가지의 질문이 있으며,**Q1& Q6은 모든 input이 거치게 되는 질문**이 됨. 이중 **Q1는 가장 첫 질문이 되며,Q6은 이미 분류과정을 거친 이후에 최종적으로 odd(이상현상여부)을 판별**하기 위함

- Q1(최상위)에 대한 가능한 응답은 A11,A12,A13이며 이들의 합은 확률로써 1이 됨. 
```
A11+A12+A13 =1 
A61+A62=1 (Q7,Q9,Q5와 이들의 하위 질문인 Q6사이에는 연관관계가 존재하지 않음)
```

- Q2[<-A12]의 하위 질문인 Q3,Q4,Q9은 각각 독립적으로 존재하여서 Q2에 종속됨
```
A31+A32=A22 
A41+A42=A22
A91+A92+A93=A21
(A21+A22=A12)
```

- Q4의 하위 질문인 Q10[<-A41],Q11는 Q5[<-A42]로 다시 돌아오기 때문에 결국 A5의 전체 합은 A42가 아닌 Q4(A41+A42=A22)의 상위인 Q2인 A22가 됨
```
A31+A32=A22
A41+A42=A22
A101+...+A103=A41
A111+...+A116=A41
A51+...+A54=A22
```
- Q8은 Q6의 A61에 대한 것이기 때문에 A81+...A87=A61

- 가장 하위 단계의 **leap node**는 **Q7[A71..A73]**,**Q9[A91..A93]**, **Q5[A51..A54]**의 answer와 **A13**이 됨
```
[A11=A71+....+A73]+{A12= [A21=A91+...+A93]+[A22=A51+...+A54]} +A13 =1
```
```
raw # in training data file : 4,11,12,13,14,17,18,19 27,28,29
```


## Reference
- [The Galaxy Zoo Decision Tree](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/details/the-galaxy-zoo-decision-tree)
-  [Galaxy Zoo 2: detailed morphological classifications for 304,122 galaxies from the Sloan Digital Sky Survey](https://arxiv.org/pdf/1308.3496.pdf)
