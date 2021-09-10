import numpy as np
import torch 

from collections import defaultdict
from torch import nn
from torch.nn.functional import softmax
from scipy import special
from scipy.optimize import minimize
from scipy.stats import rankdata

from coordinate_ot_distance import coordinate_ot_dist
from feature_selection_acc import get_quantile, get_ht, get_delta_stable_points, get_hts
from sklearn.model_selection import train_test_split
from metric import performance_logloss


def get_quantile(pred):
    return rankdata(pred, "average") / len(pred)

def get_ht(quantile, pred_ref):
    return np.quantile(pred_ref, quantile)


def subdomain_sample(subdomains, sublabels, n):
    data_res = []
    label_res = []
    
    for i in range(len(sublabels)):
        pos_ind = np.where(sublabels[i][:, 1] == 1)[0]
        neg_ind = np.where(sublabels[i][:, 1] == 0)[0]
        
        pos_sample = np.random.choice(pos_ind, n, replace=False)
        neg_sample = np.random.choice(neg_ind, n, replace=False)
        
        sample_ind = np.hstack([pos_sample, neg_sample])
        np.random.shuffle(sample_ind)
        
        data_res.append(subdomains[i][sample_ind])
        label_res.append(sublabels[i][sample_ind])
    
    return data_res, label_res


def sub_domain_ind(data, label, spoints):
    data_res = []
    label_res = []
    
    if len(spoints) > 0:
    
        data_res.append(data[label[:,0]<spoints[0]])
        label_res.append(label[label[:,0]<spoints[0]])

        for i in range(len(spoints)-1):
            ind = np.all([ label[:,0]>=spoints[i], label[:,0]<spoints[i+1] ], axis=0)
            data_res.append(data[ind])
            label_res.append(label[ind])

        data_res.append(data[label[:,0]>=spoints[-1]])
        label_res.append(label[label[:,0]>=spoints[-1]])
    
    else:
        data_res.append(data)
        label_res.append(label)
    
    return data_res, label_res


def sub_domain(data, spoints):
    res = []
    
    if len(spoints) > 0:
    
        res.append(data[:spoints[0]])

        for i in range(len(spoints)-1):
            res.append(data[spoints[i]:spoints[i+1]])

        res.append(data[spoints[-1]:])
    else:
        res.append(data)
    
    return res


def valid_spoints(spoints, upper_bound):
    spoints = np.sort(np.array(spoints, dtype=int))

    if len(spoints) > 0:
        spoints[0] = np.max([2000, spoints[0]])
        spoints[0] = np.min([upper_bound - 2000, spoints[0]])
        
        for i in range(1, len(spoints)):
            spoints[i] = np.max([spoints[i-1] + 2000, spoints[i]])
            spoints[i] = np.min([upper_bound - 2000*(len(spoints)-i), spoints[i]])
    return spoints


def get_min_func(data, cate_dim, num_dim, lmbda=None, njobs=20):
    
    def min_func(spoints):
        spoints = valid_spoints(spoints, len(data))
        
        sub_datas = sub_domain(data, spoints)
        
        dist = []
        for i in range(len(sub_datas)-1):
            cdt = coordinate_ot_dist(sub_datas[i], sub_datas[i+1], cate_dim, num_dim, lmbda, njobs).sum()
            dist.append(cdt)
        
        return -np.sum(dist)
    
    return min_func



# nelder_mead_simplex method

def nm_minimize(data, cate_dim, num_dim, k, max_iter=100, tol=1e-2):

    spoints = np.array([int(data.shape[0] / k) * ki for ki in range(1, k)])
    nm_func = get_min_func(data, cate_dim, num_dim)
    res = minimize(nm_func, spoints, method='Nelder-Mead', tol=tol, options={"maxiter": max_iter})
    
    return valid_spoints(res.x, len(data)), res


def pred_ij(model, adapts, fmasks, data, repeat=5, njobs=20):
    predij = []
    for i in range(len(adapts)):
        predi = []
        predij.append(predi)
        for j in range(len(adapts[i])):
            data_trans = adapts[i][j].transform(data, repeat=repeat, interpolation=fmasks[i][j], njobs=njobs)
            if isinstance(model, list):
                pred = model[i][j].predict(data_trans).reshape((repeat, -1)).mean(axis=0)
            else:
                pred = model.predict(data_trans).reshape((repeat, -1)).mean(axis=0)
            predi.append(pred)
    
    return np.array(predij)


def get_pi(pscore, data):
    pi = []
    for i in range(len(pscore)):
        pi.append(pscore[i].score(data))
    
    return special.softmax(pi, axis=0)


def weight_pred(pi, predij, sij_softmax):
    A = np.hstack([(pi[i] * predij[i]).T for i in range(len(pi))])
    return A.dot(sij_softmax).reshape(-1)



def sij_sup_minimizer(A, y, A_valid, y_valid, kt, ks, max_iter, lr, tol):
    sij = torch.ones(kt, ks, requires_grad=True)
    sij.data = sij.data * 1/ks
    
    l_prev = float('inf')
    for i in range(max_iter):
        sij_prob = softmax(sij, dim=1).reshape((-1, 1))

        l = nn.BCELoss()(torch.clamp(A.mm(sij_prob), 0, 1), y[:,None])

        if l_prev - float(l.data) <= tol:
            # print("early stop:", i, flush=True)
            break

        l_prev = float(l.data)
        l.backward(retain_graph=True)

        sij.data = sij.data - lr * sij.grad
        sij.grad.zero_()
    
    return sij.detach().numpy()


# Supervised sij selection
def sij_minimization_sup(pi, predij, label, lr=0.1, max_iter=1000, n_bootstrap=500, tol=1e-3):
    """
    pi: kt * nt
    predij: kt * ks * nt
    """

    kt, ks, nt = predij.shape
    
    agg_res = []
    for nb in range(n_bootstrap):

        ind = np.random.choice(predij.shape[-1], predij.shape[-1], replace=True)

        pi_sample = pi[:, ind].copy()
        predij_sample = predij[:, :, ind].copy()
        label_sample = label[ind].copy()

        
        A = np.hstack([(pi_sample[i] * predij_sample[i]).T for i in range(kt)])
        A = torch.from_numpy(A).float()

        y = torch.from_numpy(label_sample).float()

        sij = sij_sup_minimizer(A, y, None, None, kt, ks, max_iter, lr, tol)

        agg_res.append(special.softmax(sij, axis=1))

    sij_softmax = np.mean(agg_res, axis=0)
    return (sij_softmax / sij_softmax.sum(axis=1).reshape((-1, 1))).reshape((-1, 1))


# Supervised + regularization
def sij_minimization_weakly(pi, predij, label, 
    pi_all, predij_all, source_pred_ref,
    lr=0.1, max_iter=1000, n_bootstrap=500, tol=1e-3):
    
    kt, ks, nt = predij.shape
    n = predij_all.shape[-1]

    agg_res = []
    for nb in range(n_bootstrap):
        # labeled target example
        train_ind, test_ind = train_test_split(np.arange(nt), test_size=0.25, shuffle=True)

        pi_sample = pi[:, train_ind].copy()
        predij_sample = predij[:, :, train_ind].copy()
        label_sample = label[train_ind].copy()

        A = np.hstack([(pi_sample[i] * predij_sample[i]).T for i in range(kt)])
        A = torch.from_numpy(A).float()
        y = torch.from_numpy(label_sample).float()

        pi_valid = pi[:, test_ind].copy()
        predij_valid = predij[:, :, test_ind].copy()
        label_valid = label[test_ind].copy()

        A_valid = np.hstack([(pi_valid[i] * predij_valid[i]).T for i in range(kt)])
        A_valid = torch.from_numpy(A_valid).float()
        y_valid = torch.from_numpy(label_valid).float()

        # all (unlabeled) target example
        train_ind, test_ind = train_test_split(np.arange(n), test_size=0.25, shuffle=True)

        pi_all_sample = pi_all[:, train_ind].copy()
        predij_all_sample = predij_all[:, :, train_ind].copy()

        A_all = np.hstack([(pi_all_sample[i] * predij_all_sample[i]).T for i in range(kt)])
        A_all = torch.from_numpy(A_all).float()

        pi_all_valid = pi_all[:, test_ind].copy()
        predij_all_valid = predij_all[:, :, test_ind].copy()

        A_all_valid = np.hstack([(pi_all_valid[i] * predij_all_valid[i]).T for i in range(kt)])
        A_all_valid = torch.from_numpy(A_all_valid).float()

        # initialized with supervised minimizer
        sij_numpy = sij_sup_minimizer(A, y, A_valid, y_valid, kt, ks, max_iter, lr, tol)
        sij = torch.tensor(sij_numpy, requires_grad=True, dtype=torch.float)

        l_prev = float('inf')
        for i in range(max_iter):
            sij_prob = softmax(sij, dim=1).reshape((-1, 1))

            pred = torch.clamp(A_all.mm(sij_prob), 0, 1)#.detach().numpy().reshape(-1)
            pred_valid = torch.clamp(A_all_valid.mm(sij_prob), 0, 1)#.detach().numpy().reshape(-1)

            y_all = torch.from_numpy(
                get_ht(get_quantile(pred.detach().numpy().reshape(-1)), source_pred_ref)).float()
            y_all_valid = torch.from_numpy(
                get_ht(get_quantile(pred_valid.detach().numpy().reshape(-1)), source_pred_ref)).float()

            binary_loss = nn.BCELoss()(torch.clamp(A.mm(sij_prob), 0, 1), y[:,None])
            mse_loss = nn.MSELoss()(pred, y_all[:,None])
            l = binary_loss + mse_loss

            binary_loss = nn.BCELoss()(torch.clamp(A_valid.mm(sij_prob), 0, 1), y_valid[:,None])
            mse_loss = nn.MSELoss()(pred_valid, y_all_valid[:,None])
            l_valid = binary_loss + mse_loss


            if l_prev - float(l_valid.data) <= tol:
                # print("early stop:", i, flush=True)
                break

            l_prev = float(l_valid.data)
            l.backward(retain_graph=True)

            sij.data = sij.data - lr * sij.grad
            sij.grad.zero_()

        agg_res.append(softmax(sij, dim=1).detach().numpy())

    sij_softmax = np.mean(agg_res, axis=0)
    return (sij_softmax / sij_softmax.sum(axis=1).reshape((-1, 1))).reshape((-1, 1))


# Only regularization
def sij_minimization_unsup(pi_all, predij_all, source_pred_ref,
    lr=0.1, max_iter=1000, n_bootstrap=500, tol=1e-3):

    kt, ks, n = predij_all.shape

    agg_res = []
    for nb in range(n_bootstrap):
        ind = np.random.choice(pi_all.shape[-1], pi_all.shape[-1], replace=True)

        pi_all_sample = pi_all[:, ind].copy()
        predij_all_sample = predij_all[:, :, ind].copy()

        A_all = np.hstack([(pi_all_sample[i] * predij_all_sample[i]).T for i in range(kt)])
        A_all = torch.from_numpy(A_all).float()

        # initialized with supervised minimizer
        sij = torch.ones(kt, ks, requires_grad=True)
        sij.data = sij.data * 1/ks

        l_prev = float('inf')
        for i in range(max_iter):
            sij_prob = softmax(sij, dim=1).reshape((-1, 1))

            pred = torch.clamp(A_all.mm(sij_prob), 0, 1)#.detach().numpy().reshape(-1)

            y_all = torch.from_numpy(
                get_ht(get_quantile(pred.detach().numpy().reshape(-1)), source_pred_ref)).float()

            l = nn.MSELoss()(pred, y_all[:,None])

            if l_prev - float(l.data) <= tol:
                # print("early stop:", i, flush=True)
                break

            l_prev = float(l.data)
            l.backward(retain_graph=True)

            sij.data = sij.data - lr * sij.grad
            sij.grad.zero_()

        agg_res.append(softmax(sij, dim=1).detach().numpy())

    sij_softmax = np.mean(agg_res, axis=0)
    return (sij_softmax / sij_softmax.sum(axis=1).reshape((-1, 1))).reshape((-1, 1))


def sij_minimization_unsup_pseudo():
    pass

def element_bootstrap(pred_ts, preds, label, hts_pred_diff, perf_func, lmbda=1):

    pseudo_loss = hts_pred_diff.mean(axis=2).min(axis=0)

    perfs = np.array([perf_func(preds[i], label) for i in range(len(preds))])
    
    opt_ind = np.argmax(perfs)
    real_loss = abs(preds - preds[opt_ind]).mean(axis=1)

    ind = np.argmin(real_loss + lmbda * pseudo_loss)

    i = 0
    for best_kt in pred_ts:
        for best_ks in pred_ts[best_kt]:
            i += 1
            if i > ind:
                break
        if i > ind:
            break
    return best_kt, best_ks


def element_bootstrap_unsup(pred_ts, hts_pred_diff):
    pseudo_loss = hts_pred_diff.mean(axis=2).min(axis=0)
    ind = np.argmin(pseudo_loss)

    i = 0
    for best_kt in pred_ts:
        for best_ks in pred_ts[best_kt]:
            i += 1
            if i > ind:
                break
        if i > ind:
            break
    return best_kt, best_ks


def element_optimal_ts(perf_ts):
    best_perf = -float('inf')
    best_kt = 1
    best_ks = 1
    for kt, val in perf_ts.items():
        for ks, perf in val.items():
            if perf > best_perf:
                best_perf = perf
                best_kt = kt
                best_ks = ks
    
    return best_kt, best_ks


def balance_bootstrap(label):
    pos_sample_ind = np.random.choice(np.where(label == 1)[0], (label==1).sum(), replace=True)
    neg_sample_ind = np.random.choice(np.where(label == 0)[0], (label==0).sum(), replace=True)
    return np.hstack([pos_sample_ind, neg_sample_ind])


# Supervised ks kt selection
def optimal_kts_sup(pred_ts, target_sample_label, perf_func, n_bootstrap=500):
    n = len(target_sample_label)

    count_ts = defaultdict(dict)
    for _ in range(n_bootstrap):
        ind_sample = np.random.choice(n, n, replace=True)

        perf_ts = defaultdict(dict)
        for kt in pred_ts.keys():
            for ks in pred_ts[kt].keys():
                perf_ts[kt][ks] = perf_func(pred_ts[kt][ks][ind_sample], target_sample_label[ind_sample])
        
        best_kt, best_ks = element_optimal_ts(perf_ts)
        count_ts[best_kt].setdefault(best_ks, 0)
        count_ts[best_kt][best_ks] += 1
    
    best_kt, best_ks = element_optimal_ts(count_ts)
    
    return best_kt, best_ks, count_ts


def optimal_counter_weight_sup(pred_ts, target_sample_label, n_bootstrap=500, max_iter=2000, tol=1e-3, lr=100):
    pred_all = []
    for kt in pred_ts:
        for ks in pred_ts[kt]:
            pred_all.append(pred_ts[kt][ks])
    pred_all = np.array(pred_all)


    agg_res = []
    for _ in range(n_bootstrap):

        ind = np.random.choice(pred_all.shape[-1], pred_all.shape[-1], replace=True)

        pred_train = torch.from_numpy(pred_all[:,ind]).float()

        label_train = torch.from_numpy(target_sample_label[ind]).float()

        w = np.zeros(pred_train.shape[0])
        w[0] = np.log(3*(len(pred_all)-1))
        w = torch.tensor(w, requires_grad=True, dtype=torch.float)

        l_prev = float('inf')
        for i in range(max_iter):
            w_prob = softmax(w, dim=0)

            pred = torch.clamp((pred_train.T * w_prob).sum(axis=1), 0, 1)#.detach().numpy().reshape(-1)

            l = nn.BCELoss()(pred, label_train)

            if l_prev - float(l.data) <= tol:
                print("early stop:", i, flush=True)
                break

            l_prev = float(l.data)
            l.backward(retain_graph=True)

            w.data = w.data - lr * w.grad
            # print(w.grad, flush=True)
            w.grad.zero_()
        agg_res.append(softmax(w, dim=0).detach().numpy())

    w_softmax = np.mean(agg_res, axis=0)
    w_softmax = w_softmax / w_softmax.sum()

    counter_weight = defaultdict(dict)
    i = 0
    for kt in pred_ts:
        for ks in pred_ts[kt]:
            counter_weight[kt][ks] = w_softmax[i]
            i += 1
    return counter_weight


# Supervised + pseudo label ks kt selection
def optimal_kts_weakly(pred_ts, label, sample_ind, pred_all_ts, source_pred, 
    perf_func=performance_logloss, lmbda=1, delta=0.05, n_bootstrap=20, verbose=True):
    """
    sample_ind: index of weakly labeled examples
    """

    # for unlabled data pseudo label
    pred_all = []
    for kt in pred_all_ts:
        for ks in pred_all_ts[kt]:
            pred_all.append(pred_all_ts[kt][ks])
    pred_all = np.array(pred_all)

    sample_mask = np.ones(pred_all.shape[-1])
    sample_mask[sample_ind] = 0

    pred_all = pred_all[:, sample_mask.astype(bool)]

    quantiles = [get_quantile(pred_all[i]) for i in range(len(pred_all))]
    hts = get_hts(quantiles, source_pred)
    
    instance_mask = get_delta_stable_points(hts, delta)
    if verbose:
        print("Selected Percentage", instance_mask.mean())

    pred_stable = pred_all[:, instance_mask]
    hts_stable = hts[:, instance_mask]

    hts_pred_diff = abs(np.array([pred_stable - hts_stable[i] for i in range(hts_stable.shape[0])]))

    # for weakly labeld data labels
    preds = []
    for kt in pred_ts:
        for ks in pred_ts[kt]:
            preds.append(pred_ts[kt][ks])
    preds = np.array(preds)


    count_ts = defaultdict(dict)
    for _ in range(n_bootstrap):
        ind_label_sample = balance_bootstrap(label)
        ind_pseudo_sample = np.random.choice(hts_pred_diff.shape[-1], hts_pred_diff.shape[-1], replace=True)

        best_kt, best_ks = element_bootstrap(
            pred_ts, preds[:, ind_label_sample], label[ind_label_sample], 
            hts_pred_diff[:,:,ind_pseudo_sample], perf_func, lmbda)
            
        count_ts[best_kt].setdefault(best_ks, 0)
        count_ts[best_kt][best_ks] += 1

    best_kt, best_ks = element_optimal_ts(count_ts)
    
    return best_kt, best_ks, count_ts


# Only psuedo label ks kt selection
def optimal_kts_unsup(pred_all_ts, source_pred, delta=0.05, n_bootstrap=20, verbose=True):
    """
    sample_ind: index of weakly labeled examples
    """

    # for unlabled data pseudo label
    pred_all = []
    for kt in pred_all_ts:
        for ks in pred_all_ts[kt]:
            pred_all.append(pred_all_ts[kt][ks])
    pred_all = np.array(pred_all)

    quantiles = [get_quantile(pred_all[i]) for i in range(len(pred_all))]
    hts = get_hts(quantiles, source_pred)
    
    instance_mask = get_delta_stable_points(hts, delta)
    if verbose:
        print("Selected Percentage", instance_mask.mean())

    pred_stable = pred_all[:, instance_mask]
    hts_stable = hts[:, instance_mask]

    hts_pred_diff = abs(np.array([pred_stable - hts_stable[i] for i in range(hts_stable.shape[0])]))

    count_ts = defaultdict(dict)
    for _ in range(n_bootstrap):
        ind_pseudo_sample = np.random.choice(hts_pred_diff.shape[-1], hts_pred_diff.shape[-1], replace=True)

        best_kt, best_ks = element_bootstrap_unsup(
            pred_all_ts, hts_pred_diff[:,:,ind_pseudo_sample])

        count_ts[best_kt].setdefault(best_ks, 0)
        count_ts[best_kt][best_ks] += 1

    best_kt, best_ks = element_optimal_ts(count_ts)
    
    return best_kt, best_ks, count_ts


def optimal_kts_unsup_reg(pred_all_ts, source_pred, n_bootstrap=20, max_iter=100, tol=1e-3, lr=1):

    # for unlabled data pseudo label
    pred_all = []
    for kt in pred_all_ts:
        for ks in pred_all_ts[kt]:
            pred_all.append(pred_all_ts[kt][ks])
    pred_all = np.array(pred_all)


    agg_res = []
    for _ in range(n_bootstrap):

        ind = np.random.choice(pred_all.shape[-1], pred_all.shape[-1], replace=True)


        pred_train = torch.from_numpy(pred_all[:, ind]).float()
        w = np.zeros(pred_train.shape[0])
        w[0] = np.log(3*(len(pred_all)-1))
        w = torch.tensor(w, requires_grad=True, dtype=torch.float)

        l_prev = float('inf')
        for i in range(max_iter):
            w_prob = softmax(w, dim=0)

            pred = torch.clamp((pred_train.T * w_prob).sum(axis=1), 0, 1)#.detach().numpy().reshape(-1)

            y_all = torch.from_numpy(
                get_ht(get_quantile(pred.detach().numpy().reshape(-1)), source_pred)).float()

            l = nn.MSELoss()(pred, y_all)

            if l_prev - float(l.data) <= tol:
                print("early stop:", i, flush=True)
                break

            l_prev = float(l.data)
            l.backward(retain_graph=True)


            w.data = w.data - lr * w.grad
            w.grad.zero_()
        agg_res.append(softmax(w, dim=0).detach().numpy())

    w_softmax = np.mean(agg_res, axis=0)
    w_softmax = w_softmax / w_softmax.sum()

    counter_weight = defaultdict(dict)
    i = 0
    for kt in pred_all_ts:
        for ks in pred_all_ts[kt]:
            counter_weight[kt][ks] = w_softmax[i]
            i += 1
    return counter_weight



def counter_to_weight(vote_counter):
    c = 0
    for kt in vote_counter:
        for ks in vote_counter[kt]:
            c += vote_counter[kt][ks]
    
    weight = defaultdict(dict)
    for kt in vote_counter:
        for ks in vote_counter[kt]:
            weight[kt][ks] = vote_counter[kt][ks] / c
    
    return weight
