#!/usr/bin/env python
# coding: utf-8

# ### NN

# In[ ]:


import sys
import warnings
sys.path.append("../src/")
sys.path.append("../model/")
warnings.filterwarnings('ignore')

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from nn_model import fully_connected_embed, embed_nn
from io_utils import load_dataset, save_model, model_log, load_model, load_pickle, save_pickle
from metric import performance_pr_auc, performance_logloss

# Constant
task = "kaggle" # the dataset that we are working on
data_type = "uni" # the type of data that we are dealing with
model_type = "nn"
version = "uni" # the version of prediction model
num_dim = 43

epoch = 25
batch_size = 1024
period = [0, 1, 2] # the period of data
cate_index = 8 # the index of the last categorical feature

device = torch.device("cuda") # device of training 
        
embedding_input = [3, 131, 4, 483, 103, 5, 106, 4] # different levels of categorical features
embedding_dim = [1, 3, 1, 4, 3, 1, 3, 1] # embedding dimension

# Train the model
domain = "source"
models = []
perfs = []
for seed in range(10):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Repetation", task, domain, seed, flush=True)
    train, train_label, test, test_label = load_dataset("../data/", task, domain, data_type, 0)

    # Train the model with the best learning rate
    train, valid, train_label, valid_label = train_test_split(train, train_label, test_size=0.25, 
                                                              shuffle=True, random_state=seed)
    embed = embed_nn(embedding_input, embedding_dim, num_dim)
    model = fully_connected_embed(embed, cate_index, device)

    model.fit(train, train_label[:, 1], 
              train, 
              valid, valid_label[:, 1], 
              epoch=epoch, batch_size=batch_size, lr=0.005, beta=0, 
              early_stop=False, verbose=False)

    models.append(model)
    pred = model.predict(test)
    perf = performance_logloss(pred, test_label[:, 1])
    perfs.append(perf)

model = models[np.argmax(perfs)]
# Save prediction model
save_model(model, "../model/", task, domain, model_type, 0, version)

print("Done")


# ### LGB

# In[ ]:


import sys
import warnings
sys.path.append("../src/")
sys.path.append("../model/")
warnings.filterwarnings('ignore')

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold

from io_utils import load_dataset, save_model, model_log, load_model, load_pickle
from metric import performance_pr_auc, performance_logloss

# Constant
task = "kaggle" # the dataset that we are working on
data_type = "uni" # the type of data that we are dealing with
model_type = "lgb"
period = [0, 1, 2] # the period of data
cate_index = 8 # the index of the last categorical feature
version = "uni" # the version of embedding matrix & prediction model


params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.04,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'num_threads': 30
}


# Train the model
domain = "source"
models = []
perfs = []
for seed in range(10):
    np.random.seed(seed)
    params["random_state"] = seed

    print("Repetation", task, domain, seed, flush=True)
    train, train_label, test, test_label = load_dataset("../data/", task, domain, data_type, 0)

    # Train the model with the best learning rate
    train, valid, train_label, valid_label = train_test_split(train, train_label, test_size=0.25, 
                                                              shuffle=True, random_state=seed)
    
    lgb_train = lgb.Dataset(train, train_label[:,1], categorical_feature=range(cate_index))
    lgb_valid = lgb.Dataset(valid, valid_label[:,1], categorical_feature=range(cate_index))

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=5000, 
                    early_stopping_rounds=20, verbose_eval=None)

    models.append(model)
    pred = model.predict(test)
    perf = performance_logloss(pred, test_label[:, 1])
    perfs.append(perf)

model = models[np.argmax(perfs)]
# Save prediction model
save_model(model, "../model/", task, domain, model_type, 0, version)

print("Done")


# ### Baseline Performance

# In[ ]:


import sys
sys.path.append("../src/")
sys.path.append("../model/")

import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold

from io_utils import load_dataset, save_model, model_log, load_model, load_pickle
from metric import performance_pr_auc, performance_logloss

# Constant
task = "kaggle" # the dataset that we are working on
data_type = "uni" # the type of data that we are dealing with
period = [0, 1, 2] # the period of data
version = "uni" # the version of embedding matrix & prediction model

model_type = "nn"

# Logloss
for model_domain in ["source"]:
    for model in models:
#     model = load_model("../model/", task, model_domain, model_type, 0, version)
        train, train_label, test, test_label = load_dataset("../data/", task, "source", data_type, 0)

        pred = model.predict(test)

        perf = performance_logloss(pred, test_label[:, 1])
        model_log("../logs/logloss", task, model_domain, model_type, 0, version, "{}: {}".format("source", perf))

        for p in period:
            train, train_label, test, test_label = load_dataset("../data/", task, "target", data_type, p)

            pred = model.predict(test)

            perf = performance_logloss(pred, test_label[:, 1])
            model_log("../logs/logloss", task, model_domain, model_type, p, version, "{}: {}".format("target", perf))

        
for model_domain in ["source"]:
    for model in models:
#     model = load_model("../model/", task, model_domain, model_type, 0, version)
        train, train_label, test, test_label = load_dataset("../data/", task, "source", data_type, 0)

        pred = model.predict(test)

        perf = performance_pr_auc(pred, test_label[:, 1])
        model_log("../logs/pr_auc", task, model_domain, model_type, 0, version, "{}: {}".format("source", perf))

        for p in period:
            train, train_label, test, test_label = load_dataset("../data/", task, "target", data_type, p)

            pred = model.predict(test)

            perf = performance_pr_auc(pred, test_label[:, 1])
            model_log("../logs/pr_auc", task, model_domain, model_type, p, version, "{}: {}".format("target", perf))


            
model_type = "lgb"

# Logloss
for model_domain in ["source"]:
    for model in models:
#     model = load_model("../model/", task, model_domain, model_type, 0, version)
        train, train_label, test, test_label = load_dataset("../data/", task, "source", data_type, 0)

        pred = model.predict(test)

        perf = performance_logloss(pred, test_label[:, 1])
        model_log("../logs/logloss", task, model_domain, model_type, 0, version, "{}: {}".format("source", perf))

        for p in period:
            train, train_label, test, test_label = load_dataset("../data/", task, "target", data_type, p)

            pred = model.predict(test)

            perf = performance_logloss(pred, test_label[:, 1])
            model_log("../logs/logloss", task, model_domain, model_type, p, version, "{}: {}".format("target", perf))

        
for model_domain in ["source"]:
    for model in models:
#     model = load_model("../model/", task, model_domain, model_type, 0, version)
        train, train_label, test, test_label = load_dataset("../data/", task, "source", data_type, 0)

        pred = model.predict(test)

        perf = performance_pr_auc(pred, test_label[:, 1])
        model_log("../logs/pr_auc", task, model_domain, model_type, 0, version, "{}: {}".format("source", perf))

        for p in period:
            train, train_label, test, test_label = load_dataset("../data/", task, "target", data_type, p)

            pred = model.predict(test)

            perf = performance_pr_auc(pred, test_label[:, 1])
            model_log("../logs/pr_auc", task, model_domain, model_type, p, version, "{}: {}".format("target", perf))

print("Done")


# In[ ]:




