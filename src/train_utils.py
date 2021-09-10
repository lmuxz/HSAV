from os import replace
import time
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import linprog

def extend_dataset(xs, xt, shuffle=False):
    ns = xs.shape[0]
    nt = xt.shape[0]
    n = max(ns, nt)
    indexs = np.random.choice(range(ns), n, replace=True)
    indext = np.random.choice(range(nt), n, replace=True)
    return_s = list(range(ns)) + list(indexs[ns:n])
    return_t = list(range(nt)) + list(indext[nt:n])

    np.random.shuffle(return_s)
    np.random.shuffle(return_t)
    return return_s, return_t


def reduce_dataset(xs, xt, shuffle=False):
    ns = xs.shape[0]
    nt = xt.shape[0]

    if ns > nt:
        indexs = np.random.choice(ns, nt, replace=False)
        indext = np.arange(nt)
    else:
        indexs = np.arange(ns)
        indext = np.random.choice(nt, ns, replace=False)
    
    np.random.shuffle(indexs)
    np.random.shuffle(indext)
    return indexs, indext


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000), flush=True)
        return result
    return timed



def sample_validation_data(task, label, ratio=0.05, number_examples=None):
    if number_examples is None:
        number_examples = int(label.shape[0] * ratio)
    
    pos_example = int(number_examples * label[:,1].mean()) + 1
    neg_example = number_examples - pos_example

    pos_sample_ind = np.random.choice(np.where(label[:, 1]==1)[0], pos_example, replace=False)
    neg_sample_ind = np.random.choice(np.where(label[:, 1]==0)[0], neg_example, replace=False)

    sample_index = np.sort(np.hstack([pos_sample_ind, neg_sample_ind]))
    label = label[sample_index, 1]
    return sample_index, label



def balance_sample(label, n_examples, ratio_pos=None, *args, **kwargs):
    if ratio_pos is None:
        ratio_pos = label[:,1].mean()

    n_pos = int(ratio_pos * n_examples)
    n_neg = n_examples - n_pos

    index_pos = np.random.choice(np.where(label[:, 1]==1)[0], n_pos, replace=False)
    index_neg = np.random.choice(np.where(label[:, 1]==0)[0], n_neg, replace=False)

    index = np.r_[index_pos, index_neg]
    np.random.shuffle(index)
    return index


def joint_dist_embedding(x, y):
    return np.transpose(np.matmul(x[:, :, np.newaxis], y[:, np.newaxis, :]), (0, 2, 1)).reshape(x.shape[0], -1)

def reverse_dist_embedding(x, y):
    return (x.reshape(x.shape[0], y.shape[-1], -1) / y[:, :, np.newaxis]).mean(axis=1)


def balance_clustering(affinity, feature_index=None, n=2):
    if feature_index is None:
        feature_index = np.arange(len(affinity))

    if len(feature_index) > n:
        spc = SpectralClustering(n_clusters=2, affinity="precomputed")
        spc.fit(affinity)
        index_cluster1 = np.where(spc.labels_==0)[0]
        index_cluster2 = np.where(spc.labels_==1)[0]

        affinity_left = affinity[index_cluster1[:, np.newaxis], index_cluster1]
        feature_index_left = feature_index[index_cluster1]
        feature_clusters = balance_clustering(affinity_left, feature_index_left, n)

        affinity_right = affinity[index_cluster2[:, np.newaxis], index_cluster2]
        feature_index_right = feature_index[index_cluster2]
        feature_clusters += balance_clustering(affinity_right, feature_index_right, n)
    else:
        return [feature_index]
    
    return feature_clusters


class LogScore():
    def __init__(self, cate_index):
        self.cate_index = cate_index

    def fit(self, data):
        self.eps = 1 / (2 * data.shape[0])
        self.modals = []
        self.density_estimator = []


        for i in range(data.shape[-1]):
            modality, counts = np.unique(data[:, i], return_counts=True)
            counts = counts / counts.sum()

            self.modals.append(modality)
            self.density_estimator.append({modality[i]: counts[i] for i in range(len(counts))})
    
    def score(self, data):
        data = data.copy()

        for i in range(self.cate_index, data.shape[-1]):
            data[data[:, i] < self.modals[i][0], i] = self.modals[i][0]
        
            modality_index = np.digitize(data[:, i], self.modals[i]) - 1
            data[:, i] = self.modals[i][modality_index]
        
        scores = []
        for j in range(data.shape[0]):
            score = 0
            for i in range(data.shape[-1]):
                score += np.log(self.density_estimator[i].get(data[j][i], self.eps))
            scores.append(score)

        return np.array(scores) 


def get_optimal_r(data, cate_index, split_ratio, sample_size=2000):
    log_prob_score = []
    sample_index = np.random.choice(data.shape[0], sample_size, replace=False)


    split_score = []
    for r in split_ratio:
        split_point = int(data.shape[0] * r)

        logscores = []
        logscores.append(LogScore(cate_index))
        logscores.append(LogScore(cate_index))
        log_prob_score.append(logscores)

        logscores[0].fit(data[:split_point])
        logscores[1].fit(data[split_point:])

        scores = []
        for i in range(len(logscores)):
            scores.append(logscores[i].score(data[sample_index]))
        scores = np.array(scores)

        split_score.append(
            (scores[0, sample_index < split_point] > scores[1, sample_index < split_point]).sum() + 
            (scores[0, sample_index > split_point] < scores[1, sample_index > split_point]).sum()
        )

    opt_ind = np.argmax(split_score)

    return split_ratio[opt_ind], log_prob_score[opt_ind]
