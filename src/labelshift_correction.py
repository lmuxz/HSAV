# Ajust label shift by introducing the pivot domain s'
import copy
import numpy as np
from sklearn.model_selection import train_test_split


def build_pivot_dataset(source, source_label, target_factor, source_factor=None):
    """
    Inputs:
        source_factor: ratio n_non_fraud / n_fraud
        target_factor: ratio n_non_fraud / n_fraud
    """
    non_fraud_n = (source_label[:, 1]==0).sum()
    if source_factor is None:
        source_factor = non_fraud_n / source_label[:, 1].sum()
    
    ratio = target_factor / source_factor
    if ratio > 1:
        replace = True
    else:
        replace = False
    
    fraud_index = np.where(source_label[:, 1]==1)[0]
    genuine_index = np.where(source_label[:, 1]==0)[0]

    genuine_index = np.random.choice(genuine_index, int(non_fraud_n*ratio), replace=replace)
    ind = np.sort(np.hstack([fraud_index, genuine_index]))
    return source[ind], source_label[ind], ind


def adjust_model(model, target_factor, source_factor):
    """
    Adjusting classifier to correct label shift
    """
    source_fraud_ratio = 1 / (source_factor + 1) 
    target_fraud_ratio = 1 / (target_factor + 1)
    w1 = target_fraud_ratio / source_fraud_ratio
    w0 = (1-target_fraud_ratio) / (1-source_fraud_ratio)

    model.pred_tmp = model.predict

    def predict(target, *args, **kwargs):
        pred = model.pred_tmp(target, *args, **kwargs)
        return pred * w1 / (pred * w1 + (1-pred) * w0) 
    
    model.predict = predict
    return model
