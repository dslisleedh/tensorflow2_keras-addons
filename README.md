# tensorflow2_keras-addons

## constraints  

### ClipValue [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/41b362c37759bdc90ea0d3c17e6876873df7f3b9/constraint.py#L4)  

## layers  

### DropconnectDense [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/41b362c37759bdc90ea0d3c17e6876873df7f3b9/layers.py#L4)  

 기존의 Dropout은 activation에서 일부를 0으로 만든 후 다음 층으로 넘겨줬지만, Dropconnect는 Weight와 bias에 dropout을 하는 것임.  
 
 |Dropout|Dropconnect|
 |--|--|
 |![do](https://i.stack.imgur.com/CewjH.png)|![dc](https://i.stack.imgur.com/D1QC7.png)|  
 
### AdaIn(Adaptive Instance Normalization) [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/e0e0261092504bd214c7d832b359b9d18af59db9/layers.py#L28)  

### ReflectPadding2D [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/b788e2865a956bd9a44ec86322e1661c02872179/layers.py#L47)  

### PixelNormalization [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/b482ceee00303c52069e0e83166fce98a25efaa6/layers.py#L69)

## losses  

### GAN Loss [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/41b362c37759bdc90ea0d3c17e6876873df7f3b9/losses.py#L4)  

### Least-sqaure Loss [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/41b362c37759bdc90ea0d3c17e6876873df7f3b9/losses.py#L16)  

### Wasserstein Loss [[Code]](https://github.com/dslisleedh/tensorflow2_keras-addons/blob/41b362c37759bdc90ea0d3c17e6876873df7f3b9/losses.py#L28)  
