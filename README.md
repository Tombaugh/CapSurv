# CapSurv
code for 《CapSurv: Capsule Network for Survival Analysis With Whole Slide Pathological Images》

# Environment

python 2.7  Keras 2.1.4 tensorflow 1.5.0

GPU: 2*1080Ti

# Data
train.npy: training data that **must be ranked from long to short according to survival time**

validation.npy: validation data

test.npy: test data

train_label.npy: the survival time of training data

validation_label.npy: the survival time of validation data

test_label.npy: the survival time of test data

train_label_onehot.npy: the one hot encoding of long or short term survivors of training data. The patients with no longer than 1-year survival are categorized as short term survivors labeled as 0, then the others as long term survivors labeled as 1

validation_label_onehot.npy: the one hot encoding of long or short term survivors of validation data

test_label_onehot.npy: the one hot encoding of long or short term survivors of test data

# Usage
Use below instruction to run the code

```
python CapSurv.py
```
