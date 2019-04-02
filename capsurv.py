# -*- coding:utf-8 -*-
__author__ = 'Tombaugh'

import sys
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
import keras
import numpy as np
#from keras_tqdm import TQDMNotebookCallback
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from keras.callbacks import TensorBoard,ModelCheckpoint
import os
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from capsulenet import CapsNet, margin_loss
from utils import combine_images
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

save_dir = '/home/tangbo/project/gbm/CapsNet-Keras-master/github/CapSurv/'  #save path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
weights_dir = save_dir+'weights/'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)



def capsulenet():
    model, eval_model, manipulate_model = CapsNet(input_shape=(128,128,3),
                                                      n_class=2,
                                                      routings=3)
    model = multi_gpu_model(model, gpus=2)
    return model, eval_model, manipulate_model

def load_data():
    """
        load data.
        path: path of data
        train.npy: training data that must be ranked from long to short according to survival time
        validation.npy: validation data
        test.npy: test data
        train_label.npy: the survival time of training data
        validation_label.npy: the survival time of validation data
        test_label.npy: the survival time of test data
        train_label_onehot.npy: the one hot encoding of long or short term survivors of training data
                                The patients with no longer than 1-year survival are categorized as short term survivors labeled as 0,
                                then the others as long term survivors labeled as 1
        validation_label_onehot.npy: the one hot encoding of long or short term survivors of validation data
        test_label_onehot.npy: the one hot encoding of long or short term survivors of test data
    """
    path = '/home/tangbo/project/gbm/cluster/npy_hsv/classfication/sorted/4'
    os.chdir(path)
    x_train = np.load('train.npy')
    x_val = np.load('validation.npy')
    x_test = np.load('test.npy')
    y_train = np.load('train_label.npy')
    y_val = np.load('validation_label.npy')
    y_test = np.load('test_label.npy')
    y_train_onehot = np.load('train_label_onehot.npy')
    y_val_onehot = np.load('validation_label_onehot.npy')
    y_test_onehot = np.load('test_label_onehot.npy')
    return x_train,x_val,x_test,y_train,y_val,y_test,y_train_onehot,y_val_onehot,y_test_onehot

def p2p(y_true,y_predict,y_true_onehot):    #patch to patient
    y_true_onehot = np.argmax(y_true_onehot,1)
    #y_predict_onehot = np.argmax(y_predict,1)
    y_predict = y_predict[:,1]
    if len(y_true_onehot) != len(y_predict):
        print('patch_size_error')
        os._exit()
    y_true_patient = []
    y_true_patient_onehot = []
    y_predict_patient = []
    y_predict_patient_onehot = []
    cache = []
    for i in range(len(y_true)):
        if i == 0:
            y_true_patient_onehot.append(y_true_onehot[i])
            y_true_patient.append(y_true[i])
            cache.append(y_predict[i])
            continue
        if i == len(y_true)-1:
            if y_true[i] == y_true[i-1]:
                cache.append(y_predict[i])
                cache_np = np.array(cache)
                if np.sum((cache_np>=0.5)) >= np.sum((cache_np<0.5)):
                    y_predict_patient_onehot.append(1)
                else:
                    y_predict_patient_onehot.append(0)
                mean = np.mean(cache_np)
                y_predict_patient.append(mean)
                continue
            else:
                y_true_patient_onehot.append(y_true_onehot[i])
                y_true_patient.append(y_true[i])
                cache_np = np.array(cache)
                if np.sum((cache_np>=0.5)) >= np.sum((cache_np<0.5)):
                    y_predict_patient_onehot.append(1)
                else:
                    y_predict_patient_onehot.append(0)
                mean = np.mean(cache_np)
                y_predict_patient.append(mean)
                cache = []
                if y_predict[i] >= 0.5:
                    y_predict_patient_onehot.append(1)
                else:
                    y_predict_patient_onehot.append(0)
                y_predict_patient.append(y_predict[i])
                continue
        if y_true[i] == y_true[i-1]:
            cache.append(y_predict[i])
        else:
            y_true_patient_onehot.append(y_true_onehot[i])
            y_true_patient.append(y_true[i])
            cache_np = np.array(cache)
            if np.sum((cache_np>=0.5)) >= np.sum((cache_np<0.5)):
                y_predict_patient_onehot.append(1)
            else:
                y_predict_patient_onehot.append(0)
            mean = np.mean(cache_np)
            y_predict_patient.append(mean)
            cache = []
            cache.append(y_predict[i])
    if len(y_true_patient) != len(y_predict_patient):
        print('patient_size_error')
        os._exit()
    y_true_patient = np.array(y_true_patient)
    y_true_patient_onehot = np.array(y_true_patient_onehot)
    y_predict_patient = np.array(y_predict_patient)
    y_predict_patient_onehot = np.array(y_predict_patient_onehot)
    return y_true_patient,y_true_patient_onehot,y_predict_patient,y_predict_patient_onehot

def cox_loss(y_true,y_pred):
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = y_pred - log_risk
        num_observed_events = uncensored_likelihood.get_shape().as_list()[0]
        num = max(num_observed_events,16)
        loss = -K.sum(uncensored_likelihood) / num
        return loss

def mix_loss(y_ture,y_pred):
    cox_loss_weight = 0.3
    cox = cox_loss(y_ture,y_pred[:,1])
    margin = margin_loss(y_ture,y_pred)
    loss = cox*cox_loss_weight + margin*(1-cox_loss_weight)
    return loss

def cnn_test(x_train,x_val,x_test,y_train,y_val,y_test,y_train_onehot,y_val_onehot,y_test_onehot):
    model, eval_model, manipulate_model = capsulenet()
    batch = 16


    log_filepath = save_dir+'keras_log'
    tb_cb = TensorBoard(log_dir=log_filepath,batch_size=batch, histogram_freq=0)
    checkpoint = ModelCheckpoint(weights_dir + 'weights-{epoch:02d}.h5', monitor='val_loss',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    cbks = [tb_cb,checkpoint]

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)


    model.compile(optimizer = adam,
                loss=[mix_loss,'mse'],
                loss_weights=[1.,0.5],
                metrics={'capsnet': 'accuracy'})

    model.fit([x_train, y_train_onehot], [y_train_onehot, x_train],
            epochs=100,
            batch_size=batch,
            verbose=1,
            validation_data=[[x_val, y_val_onehot], [y_val_onehot, x_val]],
            #validation_split=0.1,
            shuffle=False,
            #class_weight={1:1, 0:0.5},
            callbacks=cbks)


    y_pred, x_recon = eval_model.predict(x_test, batch_size=batch)
    test_acc = float(np.sum(np.argmax(y_pred, 1) == np.argmax(y_test_onehot, 1)))/y_test_onehot.shape[0]
    auc = ROC(np.argmax(y_test_onehot, 1),y_pred[:,1])
    c_index = concordance_index(y_test,y_pred[:,1])
    y_true_patient,y_true_patient_onehot,y_pred_patient,y_pred_patient_onehot = p2p(y_test,y_pred,y_test_onehot)
    test_patient_acc = float(np.sum(y_pred_patient_onehot == y_true_patient_onehot))/y_true_patient_onehot.shape[0]
    auc_patient = ROC(y_true_patient_onehot,y_pred_patient)
    c_index_patient = concordance_index(y_true_patient,y_pred_patient)
    print('test_acc = '+str(test_patient_acc))
    print('test_auc = '+str(auc_patient))
    print('test_c_index = '+str(c_index_patient))

def ROC(label,predict):
    fpr, tpr, threshold = metrics.roc_curve(label,predict)
    auc = metrics.auc(fpr,tpr)
    return auc


def main():
    x_train,x_val,x_test,y_train,y_val,y_test,y_train_onehot,y_val_onehot,y_test_onehot = load_data()
    cnn_test(x_train,x_val,x_test,y_train,y_val,y_test,y_train_onehot,y_val_onehot,y_test_onehot)

    
if __name__ == '__main__':
    main()


