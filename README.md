# Neural Network for MNIST Classification

This repository contains a feedforward neural network implementation designed to classify handwritten digits from the MNIST dataset. The model is built using Python and NumPy, and it includes training and testing functionalities.


## Introduction
The MNIST dataset consists of 70,000 images of handwritten digits (0-9) in grayscale format. This project aims to create a neural network that can accurately classify these digits by training on a subset of the dataset.

## Dependencies
To run this project, you will need the following libraries:
- Python 3.x
- NumPy
- scikit-learn
- (Matplotlib)

## Experiments with learning rate and number of epochs

### Experiment 1:
- Epochs: 10
- Learning Rate: 1
- Results:
```python
Epoch 1/10 completed. Correctly classified 8907/10000. Accuracy: 89.07 % 
Epoch 1, Loss: 0.017605981693373452
Epoch 2/10 completed. Correctly classified 9051/10000. Accuracy: 90.51 % 
Epoch 2, Loss: 0.01502460355525308
Epoch 3/10 completed. Correctly classified 9168/10000. Accuracy: 91.68 % 
Epoch 3, Loss: 0.0134059778729235
Epoch 4/10 completed. Correctly classified 9218/10000. Accuracy: 92.18 % 
Epoch 4, Loss: 0.013154354222402501
Epoch 5/10 completed. Correctly classified 9262/10000. Accuracy: 92.62 % 
Epoch 5, Loss: 0.012476463460842515
Epoch 6/10 completed. Correctly classified 9245/10000. Accuracy: 92.45 % 
Epoch 6, Loss: 0.01247219614787174
Epoch 7/10 completed. Correctly classified 9189/10000. Accuracy: 91.89 % 
Epoch 7, Loss: 0.01343202468584501
Epoch 8/10 completed. Correctly classified 9299/10000. Accuracy: 92.99 % 
Epoch 8, Loss: 0.011900653303253989
Epoch 9/10 completed. Correctly classified 9216/10000. Accuracy: 92.16 % 
Epoch 9, Loss: 0.012684261923451672
Epoch 10/10 completed. Correctly classified 9307/10000. Accuracy: 93.07 % 
Epoch 10, Loss: 0.011555523540474141
```

###Experiment 2:
- Epochs: 20
- Learning Rate: 0.5
- Results:
```python
Epoch 1/20 completed. Correctly classified 9085/10000. Accuracy: 90.85 % 
Epoch 1, Loss: 0.014508466447337253
Epoch 2/20 completed. Correctly classified 9190/10000. Accuracy: 91.9 % 
Epoch 2, Loss: 0.012902526195423486
Epoch 3/20 completed. Correctly classified 9244/10000. Accuracy: 92.44 % 
Epoch 3, Loss: 0.012153957042845125
Epoch 4/20 completed. Correctly classified 9303/10000. Accuracy: 93.03 % 
Epoch 4, Loss: 0.01164587523816519
Epoch 5/20 completed. Correctly classified 9348/10000. Accuracy: 93.48 % 
Epoch 5, Loss: 0.011211019298687477
Epoch 6/20 completed. Correctly classified 9321/10000. Accuracy: 93.21 % 
Epoch 6, Loss: 0.011149727559415746
Epoch 7/20 completed. Correctly classified 9308/10000. Accuracy: 93.08 % 
Epoch 7, Loss: 0.011522008689125316
Epoch 8/20 completed. Correctly classified 9362/10000. Accuracy: 93.62 % 
Epoch 8, Loss: 0.010543251311626384
Epoch 9/20 completed. Correctly classified 9377/10000. Accuracy: 93.77 % 
Epoch 9, Loss: 0.010383253564521265
Epoch 10/20 completed. Correctly classified 9286/10000. Accuracy: 92.86 % 
Epoch 10, Loss: 0.011916542040525311
Epoch 11/20 completed. Correctly classified 9331/10000. Accuracy: 93.31 % 
Epoch 11, Loss: 0.011131790642451398
Epoch 12/20 completed. Correctly classified 9392/10000. Accuracy: 93.92 % 
Epoch 12, Loss: 0.010290431123847078
Epoch 13/20 completed. Correctly classified 9372/10000. Accuracy: 93.72 % 
Epoch 13, Loss: 0.010605674127212724
Epoch 14/20 completed. Correctly classified 9393/10000. Accuracy: 93.93 % 
Epoch 14, Loss: 0.010252080983300656
Epoch 15/20 completed. Correctly classified 9369/10000. Accuracy: 93.69 % 
Epoch 15, Loss: 0.010575572137749225
Epoch 16/20 completed. Correctly classified 9383/10000. Accuracy: 93.83 % 
Epoch 16, Loss: 0.010374584419232296
Epoch 17/20 completed. Correctly classified 9369/10000. Accuracy: 93.69 % 
Epoch 17, Loss: 0.010581825571705344
Epoch 18/20 completed. Correctly classified 9410/10000. Accuracy: 94.1 % 
Epoch 18, Loss: 0.010096903424236268
Epoch 19/20 completed. Correctly classified 9386/10000. Accuracy: 93.86 % 
Epoch 19, Loss: 0.010188183587423086
Epoch 20/20 completed. Correctly classified 9407/10000. Accuracy: 94.07 % 
Epoch 20, Loss: 0.009912916641316287
```

###Experiment 3:
- Epochs: 5
- Learning Rate: 2
- Results:
```python
Epoch 1/5 completed. Correctly classified 8553/10000. Accuracy: 85.53 % 
Epoch 1, Loss: 0.02429639222064912
Epoch 2/5 completed. Correctly classified 8689/10000. Accuracy: 86.89 % 
Epoch 2, Loss: 0.02204438612451149
Epoch 3/5 completed. Correctly classified 9022/10000. Accuracy: 90.22 % 
Epoch 3, Loss: 0.01761010369864547
Epoch 4/5 completed. Correctly classified 8946/10000. Accuracy: 89.46 % 
Epoch 4, Loss: 0.01862393516713422
Epoch 5/5 completed. Correctly classified 9064/10000. Accuracy: 90.64 % 
Epoch 5, Loss: 0.016553119977689656
```

###Experiment 4:
- Epochs: 20
- Learning Rate: 2
- Results:
```python
Epoch 1/20 completed. Correctly classified 8709/10000. Accuracy: 87.09 % 
Epoch 1, Loss: 0.02082964197714403
Epoch 2/20 completed. Correctly classified 8871/10000. Accuracy: 88.71 % 
Epoch 2, Loss: 0.01981819219203677
Epoch 3/20 completed. Correctly classified 8840/10000. Accuracy: 88.4 % 
Epoch 3, Loss: 0.01966719722219742
Epoch 4/20 completed. Correctly classified 8977/10000. Accuracy: 89.77 % 
Epoch 4, Loss: 0.01813737497712147
Epoch 5/20 completed. Correctly classified 9033/10000. Accuracy: 90.33 % 
Epoch 5, Loss: 0.017034969304401362
Epoch 6/20 completed. Correctly classified 9086/10000. Accuracy: 90.86 % 
Epoch 6, Loss: 0.01627758069747813
Epoch 7/20 completed. Correctly classified 9097/10000. Accuracy: 90.97 % 
Epoch 7, Loss: 0.015739522017612376
Epoch 8/20 completed. Correctly classified 9046/10000. Accuracy: 90.46 % 
Epoch 8, Loss: 0.01665031011186857
Epoch 9/20 completed. Correctly classified 9111/10000. Accuracy: 91.11 % 
Epoch 9, Loss: 0.01482499972126524
Epoch 10/20 completed. Correctly classified 9210/10000. Accuracy: 92.1 % 
Epoch 10, Loss: 0.013746764720183768
Epoch 11/20 completed. Correctly classified 9111/10000. Accuracy: 91.11 % 
Epoch 11, Loss: 0.01527509452699652
Epoch 12/20 completed. Correctly classified 9158/10000. Accuracy: 91.58 % 
Epoch 12, Loss: 0.01453821312641052
Epoch 13/20 completed. Correctly classified 9143/10000. Accuracy: 91.43 % 
Epoch 13, Loss: 0.01475132166040763
Epoch 14/20 completed. Correctly classified 9148/10000. Accuracy: 91.48 % 
Epoch 14, Loss: 0.014381069608510625
Epoch 15/20 completed. Correctly classified 9197/10000. Accuracy: 91.97 % 
Epoch 15, Loss: 0.013816348934375609
Epoch 16/20 completed. Correctly classified 9197/10000. Accuracy: 91.97 % 
Epoch 16, Loss: 0.014223469043913127
Epoch 17/20 completed. Correctly classified 9118/10000. Accuracy: 91.18 % 
Epoch 17, Loss: 0.015706953901913782
Epoch 18/20 completed. Correctly classified 9209/10000. Accuracy: 92.09 % 
Epoch 18, Loss: 0.013955310123696761
Epoch 19/20 completed. Correctly classified 9235/10000. Accuracy: 92.35 % 
Epoch 19, Loss: 0.013234334931393093
Epoch 20/20 completed. Correctly classified 9164/10000. Accuracy: 91.64 % 
Epoch 20, Loss: 0.014869135159647663
```

###Experiment 5:
- Epochs: 5
- Learning Rate: 0.5
- Results:
```python
Epoch 1/5 completed. Correctly classified 9052/10000. Accuracy: 90.52 % 
Epoch 1, Loss: 0.01499473507716492
Epoch 2/5 completed. Correctly classified 9035/10000. Accuracy: 90.35 % 
Epoch 2, Loss: 0.015511006574705692
Epoch 3/5 completed. Correctly classified 9245/10000. Accuracy: 92.45 % 
Epoch 3, Loss: 0.012117035131248758
Epoch 4/5 completed. Correctly classified 9245/10000. Accuracy: 92.45 % 
Epoch 4, Loss: 0.011749954213095923
Epoch 5/5 completed. Correctly classified 9208/10000. Accuracy: 92.08 % 
Epoch 5, Loss: 0.012413021511690204
```