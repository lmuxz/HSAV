# forward greedy search with feature cluster
import numpy as np
import lightgbm as lgb
import torch

from train_utils import timeit

from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


def multicore_helper(args):
    func = args[0]
    args = args[1:]
    return func(*args)

# @timeit
def get_hts(quantiles, ref_label, njobs=20):
    """
    Return hts for list of quantiles
    """
    args = [[get_ht, quantiles[i], ref_label] for i in range(len(quantiles))]
    with Pool(njobs) as p:
        hts = p.map(multicore_helper, args)
    return np.array(hts)

def get_quantile(pred):
    return rankdata(pred, "average") / len(pred)

def get_ht(quantile, pred_ref):
    return np.quantile(pred_ref, quantile)


def get_delta_stable_points(preds, delta=0.01):
    """
    Pairwise delta stable between predictions in preds
    """
    masks = []
    for i in range(len(preds)):
        for j in range(len(preds)):
            masks.append(abs(preds[j] - preds[i]) <= delta)
    stable_points_mask = np.all(masks, axis=0)
    return stable_points_mask

# @timeit
def get_predictions(model, valids, repeat, masks, njobs=20):
    n = valids[0].shape[0]

    data = []
    for mask in masks:
        data.append(np.hstack([valids[int(mask[i])][:,i].reshape(-1, 1) 
            for i in range(len(mask))]))
    data = np.vstack(data)

    res = []
    preds = model.predict(data)

    for i in range(len(masks)):
        pred = preds[i*n:(i+1)*n].reshape(repeat, -1).mean(axis=0)
        res.append(pred)
    
    return np.array(res)


def get_next_iteration(best):
    masks = []
    for i in range(len(best)):
        if best[i] == 0:
            next_m = best.copy()
            next_m[i] = 1
            masks.append(next_m)
    
    return masks


def get_optimal_delta(valid, cate_dim, ht_all, tol_range, repeat, verbose):
    dis = []
    bias = []
    ht_diff = ht_all.max(axis=0) - ht_all.min(axis=0)
    for bias_tol in tol_range:
        pred_mask = get_delta_stable_points(ht_all, bias_tol)
        valid_stable = valid[np.tile(pred_mask, repeat)]

        train_data = np.vstack([valid, valid_stable])
        train_label = np.r_[np.ones(valid.shape[0]), np.zeros(valid_stable.shape[0])]

        params = {
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'binary',
            'learning_rate': 0.04,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_threads': 30,
            'scale_pos_weight': pred_mask.sum() / len(pred_mask),
            'seed': 0,
            'verbose': -1,
        }
        
        train, v_data, train_label, v_label = train_test_split(train_data, train_label, test_size=0.25, shuffle=True)
        lgb_train = lgb.Dataset(train, train_label, categorical_feature=range(cate_dim))
        lgb_valid = lgb.Dataset(v_data, v_label, categorical_feature=range(cate_dim))
        
        clf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], num_boost_round=5000, 
                        early_stopping_rounds=20, verbose_eval=False)
        
        err = (1-(clf.predict(valid)>0.5)).mean() + (clf.predict(valid_stable)>0.5).mean()
        dis.append(1-err)
        bias.append(ht_diff[pred_mask].mean())

    bias_min_ind = np.argmin(np.array(dis) + np.array(bias))
    delta = tol_range[bias_min_ind]
    if verbose:
        print("Best Delta:", delta, flush=True)
    return delta


def bootstrap_element(hts_pred_diff_example):
    # hts_pred_diff_example = hts_pred_diff_example.mean(dim=2).T
    # i_best = torch.argmin(hts_pred_diff_example.min(dim=1)[0])
    pseudo_loss = hts_pred_diff_example.mean(axis=2).min(axis=0)[0]
    i_best = torch.argmin(pseudo_loss)
    # print("Loss:",  pseudo_loss, flush=True)
    # pdb.set_trace()
    return int(i_best.cpu())


# @timeit
def bootstrap(preds_all, ht_all, instance_mask, feature_masks_len, n_bootstrap, bootstrap_tol, njobs=20):
    hts_pred_diff = []
    for i in range(ht_all.shape[0]):
        hts_pred_diff.append(abs(preds_all - ht_all[i]))
    hts_pred_diff = np.array(hts_pred_diff)
    hts_pred_diff = torch.from_numpy(hts_pred_diff).float().to(torch.device("cuda"))
    
    instance_index = np.where(instance_mask==1)[0]

    bootstrap_index = [
        np.random.choice(instance_index, len(instance_index), replace=True) for _ in range(n_bootstrap)
    ]

    stop = False
    candidates = np.zeros(feature_masks_len)
    for i in range(len(bootstrap_index)):

        i_best = bootstrap_element(hts_pred_diff[:, :, bootstrap_index[i]])

        if i_best != 0:
            candidates[i_best-1] += 1
        else:
            stop = True
            print("Early Stop", flush=True)
            break


    # accept all features that surpass the bootstrap toleration. #relate:i_accept
    # i_accepted = np.where(candidates / n_bootstrap > bootstrap_tol)[0]
    # if stop is False:
    #     stop = len(i_accepted) == 0

    # accept the most significant feature that surpasses the bootstrap toleration. #relate:i_accept
    i_accepted = np.argmax(candidates)
    stop = candidates[i_accepted] / n_bootstrap < bootstrap_tol

    return candidates, i_accepted, stop


def feature_selection(model, valid, valid_trans, repeat, ref_label,
    instance_mask=None, 
    delta=None, cate_dim=None, tol_range=None,
    n_bootstrap=200, njobs=20, bootstrap_tol=0.5, max_feature=3, verbose=False):

    """
    Feature selection function for 1d adaptation
    """
    valids = [np.tile(valid, (repeat, 1)), valid_trans]
    best = np.zeros(valid_trans.shape[-1])
    best_pred = get_predictions(model, valids, repeat, [best], njobs)[0]

    stop = False
    while not stop:
        feature_masks = get_next_iteration(best)
        if len(feature_masks) > 0:
            # Predictions
            preds = get_predictions(model, valids, repeat, feature_masks)
            preds_all = np.vstack([best_pred, preds])

            # Quantiles
            quantile_previous = get_quantile(best_pred)
            quantiles = [get_quantile(pred) for pred in preds]

            # Pseudo labels
            ht_previous = get_ht(quantile_previous, ref_label)
            hts = get_hts(quantiles, ref_label, njobs=njobs)
            ht_all = np.vstack([ht_previous, hts])

            # Delta
            if delta is None:
                delta = get_optimal_delta(valid, cate_dim, ht_all, tol_range, repeat, verbose)
            
            # Instance Mask
            pred_mask = get_delta_stable_points([ht_previous] + hts, delta)

            instance_masks = [pred_mask]
            if instance_mask is not None:
                instance_masks.append(instance_mask)
            instance_mask = np.all(instance_masks, axis=0)

            if instance_mask.sum() == 0:
                break

            # Boostrap
            candidates, i_accepted, stop = bootstrap(preds_all, ht_all, instance_mask, len(feature_masks), n_bootstrap, bootstrap_tol, njobs)

            if verbose:
                print("Instance Percentage: {:.3f}".format(instance_mask.mean()), 
                    "Votes Percentage:", candidates[i_accepted]/n_bootstrap, flush=True)
        else:
            stop = True

        if stop:
            break

        # relate:i_accept
        # best = np.any([feature_masks[i] for i in i_accepted], axis=0)
        best = feature_masks[i_accepted]
        best_pred = get_predictions(model, valids, repeat, [best], njobs)[0]

        if verbose:
            print("Selected Features Clusters:", np.where(best==1)[0], flush=True)

        if best.sum() >= max_feature:
            if verbose:
                print("Maximum number of feature exceed.", flush=True)
            break

    return best
