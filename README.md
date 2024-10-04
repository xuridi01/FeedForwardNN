# Neural Network for MNIST Classification

This repository contains a feedforward neural network implementation designed to classify handwritten digits from the MNIST dataset. The model is built using Python and NumPy, and it includes training and testing functionalities. This network is designed with mini batch gradient descent, whole data set is processed sequentially in one epoch but weights and biases are changed after mini batch.


## Introduction
The MNIST dataset consists of 70,000 images of handwritten digits (0-9) in grayscale format. This project aims to create a neural network that can accurately classify these digits by training on a subset of the dataset.

## Dependencies
To run this project, you will need the following libraries:
- Python 3.x
- NumPy
- scikit-learn
- (Matplotlib)

## Experiments with learning rate and number of epochs
- Size of mini batch is 8 for these experiments

### Experiment 1:
- Epochs: 10
- Learning Rate: 1
- Results:
```python
Epoch 1/10 completed. Correctly classified 8707/10000. Accuracy: 87.07 % 
Epoch 1, Loss: 0.020431834807744267
Epoch 2/10 completed. Correctly classified 9070/10000. Accuracy: 90.7 % 
Epoch 2, Loss: 0.015126205352906638
Epoch 3/10 completed. Correctly classified 9158/10000. Accuracy: 91.58 % 
Epoch 3, Loss: 0.013517070339832366
Epoch 4/10 completed. Correctly classified 9197/10000. Accuracy: 91.97 % 
Epoch 4, Loss: 0.012682049518899735
Epoch 5/10 completed. Correctly classified 9226/10000. Accuracy: 92.26 % 
Epoch 5, Loss: 0.012187411389877046
Epoch 6/10 completed. Correctly classified 9249/10000. Accuracy: 92.49 % 
Epoch 6, Loss: 0.011883495277369448
Epoch 7/10 completed. Correctly classified 9280/10000. Accuracy: 92.8 % 
Epoch 7, Loss: 0.011542140965277049
Epoch 8/10 completed. Correctly classified 9292/10000. Accuracy: 92.92 % 
Epoch 8, Loss: 0.01140434649455433
Epoch 9/10 completed. Correctly classified 9302/10000. Accuracy: 93.02 % 
Epoch 9, Loss: 0.011282555396195217
Epoch 10/10 completed. Correctly classified 9302/10000. Accuracy: 93.02 % 
Epoch 10, Loss: 0.011228936434245813
```

### Experiment 2:
- Epochs: 20
- Learning Rate: 0.5
- Results:
```python
Epoch 1, Loss: 0.03739422056924159
Epoch 2/20 completed. Correctly classified 8737/10000. Accuracy: 87.37 % 
Epoch 2, Loss: 0.020175157303015202
Epoch 3/20 completed. Correctly classified 8938/10000. Accuracy: 89.38 % 
Epoch 3, Loss: 0.016844200222621394
Epoch 4/20 completed. Correctly classified 9032/10000. Accuracy: 90.32 % 
Epoch 4, Loss: 0.015246363343179218
Epoch 5/20 completed. Correctly classified 9098/10000. Accuracy: 90.98 % 
Epoch 5, Loss: 0.014256661591828399
Epoch 6/20 completed. Correctly classified 9134/10000. Accuracy: 91.34 % 
Epoch 6, Loss: 0.013664022005199544
Epoch 7/20 completed. Correctly classified 9177/10000. Accuracy: 91.77 % 
Epoch 7, Loss: 0.013199258656320895
Epoch 8/20 completed. Correctly classified 9195/10000. Accuracy: 91.95 % 
Epoch 8, Loss: 0.012828230000720966
Epoch 9/20 completed. Correctly classified 9205/10000. Accuracy: 92.05 % 
Epoch 9, Loss: 0.012523247682472647
Epoch 10/20 completed. Correctly classified 9220/10000. Accuracy: 92.2 % 
Epoch 10, Loss: 0.012290568627048703
Epoch 11/20 completed. Correctly classified 9230/10000. Accuracy: 92.3 % 
Epoch 11, Loss: 0.012128800802774242
Epoch 12/20 completed. Correctly classified 9240/10000. Accuracy: 92.4 % 
Epoch 12, Loss: 0.01199596203944146
Epoch 13/20 completed. Correctly classified 9250/10000. Accuracy: 92.5 % 
Epoch 13, Loss: 0.011866159920046313
Epoch 14/20 completed. Correctly classified 9262/10000. Accuracy: 92.62 % 
Epoch 14, Loss: 0.011740477085474082
Epoch 15/20 completed. Correctly classified 9267/10000. Accuracy: 92.67 % 
Epoch 15, Loss: 0.011629489465826986
Epoch 16/20 completed. Correctly classified 9277/10000. Accuracy: 92.77 % 
Epoch 16, Loss: 0.011534691643735488
Epoch 17/20 completed. Correctly classified 9283/10000. Accuracy: 92.83 % 
Epoch 17, Loss: 0.011436997116182351
Epoch 18/20 completed. Correctly classified 9283/10000. Accuracy: 92.83 % 
Epoch 18, Loss: 0.011351996502812876
Epoch 19/20 completed. Correctly classified 9301/10000. Accuracy: 93.01 % 
Epoch 19, Loss: 0.011264387164733698
Epoch 20/20 completed. Correctly classified 9304/10000. Accuracy: 93.04 % 
Epoch 20, Loss: 0.011177305078867661
```

### Experiment 3:
- Epochs: 5
- Learning Rate: 2
- Results:
```python
Epoch 1, Loss: 0.015701496490690175
Epoch 2/5 completed. Correctly classified 9203/10000. Accuracy: 92.03 % 
Epoch 2, Loss: 0.012791847695832564
Epoch 3/5 completed. Correctly classified 9256/10000. Accuracy: 92.56 % 
Epoch 3, Loss: 0.01195895844769495
Epoch 4/5 completed. Correctly classified 9279/10000. Accuracy: 92.79 % 
Epoch 4, Loss: 0.011321651598209626
Epoch 5/5 completed. Correctly classified 9270/10000. Accuracy: 92.7 % 
Epoch 5, Loss: 0.011598422147513106
```

### Experiment 4:
- Epochs: 20
- Learning Rate: 2
- Results:
```python
Epoch 1/20 completed. Correctly classified 8953/10000. Accuracy: 89.53 % 
Epoch 1, Loss: 0.016812660202272393
Epoch 2/20 completed. Correctly classified 9147/10000. Accuracy: 91.47 % 
Epoch 2, Loss: 0.013808511911803632
Epoch 3/20 completed. Correctly classified 9235/10000. Accuracy: 92.35 % 
Epoch 3, Loss: 0.012623033560568422
Epoch 4/20 completed. Correctly classified 9256/10000. Accuracy: 92.56 % 
Epoch 4, Loss: 0.01226375478792278
Epoch 5/20 completed. Correctly classified 9282/10000. Accuracy: 92.82 % 
Epoch 5, Loss: 0.01195683965142396
Epoch 6/20 completed. Correctly classified 9324/10000. Accuracy: 93.24 % 
Epoch 6, Loss: 0.011306823468654373
Epoch 7/20 completed. Correctly classified 9339/10000. Accuracy: 93.39 % 
Epoch 7, Loss: 0.011113028551400833
Epoch 8/20 completed. Correctly classified 9314/10000. Accuracy: 93.14 % 
Epoch 8, Loss: 0.011339000090624397
Epoch 9/20 completed. Correctly classified 9343/10000. Accuracy: 93.43 % 
Epoch 9, Loss: 0.011007724900796921
Epoch 10/20 completed. Correctly classified 9344/10000. Accuracy: 93.44 % 
Epoch 10, Loss: 0.010783353604411336
Epoch 11/20 completed. Correctly classified 9352/10000. Accuracy: 93.52 % 
Epoch 11, Loss: 0.010730096470737263
Epoch 12/20 completed. Correctly classified 9364/10000. Accuracy: 93.64 % 
Epoch 12, Loss: 0.010559138249562363
Epoch 13/20 completed. Correctly classified 9348/10000. Accuracy: 93.48 % 
Epoch 13, Loss: 0.010641026473216726
Epoch 14/20 completed. Correctly classified 9359/10000. Accuracy: 93.59 % 
Epoch 14, Loss: 0.010600337584960078
Epoch 15/20 completed. Correctly classified 9348/10000. Accuracy: 93.48 % 
Epoch 15, Loss: 0.010652489948406953
Epoch 16/20 completed. Correctly classified 9374/10000. Accuracy: 93.74 % 
Epoch 16, Loss: 0.010433495498263513
Epoch 17/20 completed. Correctly classified 9338/10000. Accuracy: 93.38 % 
Epoch 17, Loss: 0.010829045548123955
Epoch 18/20 completed. Correctly classified 9372/10000. Accuracy: 93.72 % 
Epoch 18, Loss: 0.01041013800485609
Epoch 19/20 completed. Correctly classified 9395/10000. Accuracy: 93.95 % 
Epoch 19, Loss: 0.01027260898064316
Epoch 20/20 completed. Correctly classified 9397/10000. Accuracy: 93.97 % 
Epoch 20, Loss: 0.010238306225814097
```

### Experiment 5:
- Epochs: 5
- Learning Rate: 0.5
- Results:
```python
Epoch 1/5 completed. Correctly classified 8379/10000. Accuracy: 83.79 % 
Epoch 1, Loss: 0.025552576838239912
Epoch 2/5 completed. Correctly classified 8827/10000. Accuracy: 88.27 % 
Epoch 2, Loss: 0.018742025917229416
Epoch 3/5 completed. Correctly classified 8974/10000. Accuracy: 89.74 % 
Epoch 3, Loss: 0.016498686860130833
Epoch 4/5 completed. Correctly classified 9040/10000. Accuracy: 90.4 % 
Epoch 4, Loss: 0.015315481511796474
Epoch 5/5 completed. Correctly classified 9078/10000. Accuracy: 90.78 % 
Epoch 5, Loss: 0.01454969802292826
```