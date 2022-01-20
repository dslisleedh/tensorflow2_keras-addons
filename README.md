# tensorflow2_keras-addons


## layers  

### DropconnectDense  

 기존의 Dropout은 activation에서 일부를 0으로 만든 후 다음 층으로 넘겨줬지만, Dropconnect는 Weight와 bias에 dropout을 하는 것임.  
 
 |Dropout|Dropconnect|
 |--|--|
 |![do](https://i.stack.imgur.com/CewjH.png)|![dc](https://i.stack.imgur.com/D1QC7.png)|
