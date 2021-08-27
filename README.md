# image-classification-level1-16
image-classification-level1-16 created by GitHub Classroom

## 현재 찍은 최고 성능 

### Gender Model

* model :efficientnet-b0  
* train/valid split : 80/20  
* optimizer : Adam  
* lr : 0.0003  
* batch size : 64  
* augmentation : randaug+cutout  
* inputsize : 224x224  
* loss_fn : LabelSmoothing(smoothing=0.05)  
* epochs : 15  

> train Loss: 0.1371 Acc: 0.9911 F1: 0.9911258239235885  
> valid Loss: 0.1574 Acc: 0.9868 **F1: 0.9860680087572518**

### Mask Model

* model :efficientnet-b0  
* train/valid split : 80/20  
* optimizer : Adam  
* lr : 0.00006  
* batch size : 64  
* augmentation : randaug+cutout  
* inputsize : 224x224
* loss_fn : LabelSmoothing(smoothing=0.05)  
* epochs : 21 

> train Loss: 0.2192 Acc: 0.9767 F1: 0.9766713917541754  
> valid Loss: 0.1739 Acc: 0.9989 **F1: 0.9985207090460696**

### Gender & Mask Model

* model :efficientnet-b0  
* train/valid split : 80/20  
* optimizer : Adam  
* lr : 0.00006  
* batch size : 64  
* augmentation : randaug+cutout  
* inputsize : 224x224
* loss_fn : LabelSmoothing(smoothing=0.05)  
* epochs : 21 

> train Loss: 0.3211 Acc: 0.9668 F1: 0.9668728849468181  
> valid Loss: 0.2878 Acc: 0.9854 **F1: 0.9842203787970396**
