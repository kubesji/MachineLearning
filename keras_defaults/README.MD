# Keras defaults
This folder contains various solutions for default Keras datasets.
## MNIST
Simple convolution network with 95 % accuracy.
## CIFAR-10
Multilayer convolutional network reaching 82 % accuracy.
## CIFAR-100
ResNet solution with roughly 50 % accuracy. It performs better than multilayer 
convolutional network, but I feel performance could be improved.
## IMDB
Simple recurrent network with one LSTM layer. Accuracy is 85 %.
## Reuters
### Recurrent network
Similar solution as with IMDB. Accuracy is around 72 % this time.
### Transformer encoder
This solution uses token and position embedding as well as transformer encoder 
from Keras NLP package. It reaches accuracy of about 80 %.
## Boston housing
### MLP
Achieved 0.89 R2 score and average 10% error of prediction from real values. There are three massive outlier which
heavily influence this metric.
### Various regressors from scikit-learn
Tried 4 regressors - nearest neighbours, support vector machine, random forest regressor and 
gradient boosting regressor. The two best performing models are RFR and GBR with R2 score around 0.9 while kNN
has 0.82. SVM is on 0.72.
