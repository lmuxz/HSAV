# improved stochastic optimal; transport plan with interpolation
import numpy as np
from numpy.random import noncentral_chisquare
import ot as pot
import io

from contextlib import redirect_stdout
from metric import of_uni_cate, similarity_to_dissimilarity
from multiprocessing import Pool

def multicore_helper(args):
    func = args[0]
    args = args[1:]
    return func(*args)


def coordinate_ot_dist(target, source, cate_dim, num_dim, lmbda=None, njobs=20, **kwargs):
    args = []

    for i in range(cate_dim + num_dim):
        if i < cate_dim:
            args.append([cate_ot_dist, target[:,i], source[:,i], lmbda])
        else:
            args.append([num_ot_dist, target[:,i], source[:,i], lmbda])

    with Pool(njobs) as p:
        ot_dist = p.map(multicore_helper, args)
    
    return np.array(ot_dist)


def cate_ot_dist(target, source, lmbda=noncentral_chisquare, **args):
    target_modality, counts= np.unique(target, return_counts=True)
    target_density = counts / counts.sum()

    # Get source modality
    source_modality, counts = np.unique(source, return_counts=True)
    source_density = counts / counts.sum()

    # Get similarity matrix
    sim, modality = of_uni_cate(source, target)
    distance = similarity_to_dissimilarity(sim)

    # Compute transportation plan
    target_index = np.where(np.in1d(modality, target_modality))[0]
    source_index = np.where(np.in1d(modality, source_modality))[0]

    if lmbda is None:
        ot_dist = pot.emd2(target_density.tolist(), source_density.tolist(), 
                    distance[target_index][:,source_index].tolist())
    else:
        ot_dist = pot.sinkhorn2(target_density.tolist(), source_density.tolist(), 
                    distance[target_index][:,source_index].tolist(), lmbda)

    return ot_dist

def num_ot_dist(target, source, lmbda=None, **args):
    target_modality, counts= np.unique(target, return_counts=True)
    target_density = counts / counts.sum()

    # Get source modality
    source_modality, counts = np.unique(source, return_counts=True)
    source_density = counts / counts.sum()

    if lmbda is None:
        ot_dist = pot.lp.emd2_1d(target_modality, source_modality, 
            target_density.tolist(), source_density.tolist())
    else:
        # # Compute the distance
        distance = pot.dist(target_modality.reshape(-1, 1), source_modality.reshape(-1, 1))
        ot_dist = pot.sinkhorn2(target_density.tolist(), source_density.tolist(), 
                    distance.tolist(), lmbda)
    
    return ot_dist
