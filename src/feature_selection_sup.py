# forward greedy search with feature cluster
import numpy as np
import torch

from train_utils import timeit


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


def bootstrap_element(preds_all, ref_label):
    loss = []
    for i in range(preds_all.shape[0]):
        loss.append(torch.nn.BCELoss()(preds_all[i, :], ref_label))
    loss = torch.hstack(loss)
    ind_opt = int(torch.argmin(loss))
    return ind_opt


# @timeit
def bootstrap(preds_all, ref_label, n_bootstrap, bootstrap_tol, device="cuda", njobs=20):
    preds_all = torch.from_numpy(preds_all).float().to(torch.device(device))
    ref_label = torch.from_numpy(ref_label).float().to(torch.device(device))

    bootstrap_index = [
        np.random.choice(preds_all.shape[-1], preds_all.shape[-1], replace=True) for _ in range(n_bootstrap)
    ]

    stop = False
    candidates = np.zeros(preds_all.shape[0])
    for i in range(len(bootstrap_index)):

        ind_opt = bootstrap_element(preds_all[:, bootstrap_index[i]], ref_label[bootstrap_index[i]])

        candidates[ind_opt] += 1

    ind_accept = np.argmax(candidates)
        
    if ind_accept==0:
        stop = True
    else:
        stop = candidates[ind_accept] / n_bootstrap < bootstrap_tol
    return candidates[1:], ind_accept-1, stop


def feature_selection_sup(model, valid, valid_trans, repeat, ref_label,
    n_bootstrap=200, njobs=20, bootstrap_tol=0.5, max_feature=3, device="cuda", verbose=False):

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

            # Boostrap
            candidates, i_accepted, stop = bootstrap(preds_all, ref_label, n_bootstrap, bootstrap_tol, device, njobs)

            if verbose:
                print("Votes Percentage:", candidates[i_accepted]/n_bootstrap, flush=True)
        else:
            stop = True

        if stop:
            break

        # best = np.any([feature_masks[i] for i in i_accepted], axis=0)
        best = feature_masks[i_accepted]
        # best_pred = get_predictions(model, valids, repeat, [best], njobs)[0]
        best_pred = preds[i_accepted]

        if verbose:
            print("Selected Features Clusters:", np.where(best==1)[0], flush=True)

        if best.sum() >= max_feature:
            if verbose:
                print("Maximum number of feature exceed.", flush=True)
            break

    return best
