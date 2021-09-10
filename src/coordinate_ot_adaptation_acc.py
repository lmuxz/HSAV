# improved stochastic optimal; transport plan with interpolation
import numpy as np
import ot as pot
import io

from contextlib import redirect_stdout
from metric import of_uni_cate, similarity_to_dissimilarity
from multiprocessing import Pool


def multicore_helper(args):
    func, target, repeat, interpolation = args
    return func.transform(target, repeat, interpolation)

def reverse_multicore_helper(args):
    func, target, repeat, interpolation = args
    return func.reverse_transform(target, repeat, interpolation)



class adaptation():
    def __init__(self, cate_dim, num_dim):
        self.cate_dim = cate_dim
        self.num_dim = num_dim

    def fit(self, target, source, feature_mask=None, distance=None, source_weight=None, target_weight=None, lmbda=None, **kwargs):
        self.adapt = []
        if feature_mask is None:
            feature_mask = np.ones(source.shape[-1], dtype=bool)

        if distance is None:
            distance = [None] * self.cate_dim

        for i in range(self.cate_dim + self.num_dim):
            if not feature_mask[i]:
                adapt = adaptation_identity()
            elif i < self.cate_dim:
                adapt = adaptation_cate_1d()

                lmbda_cur = lmbda
                adapt_flag = False
                iter = 1
                while not adapt_flag:
                    f = io.StringIO()
                    with redirect_stdout(f):
                        adapt.fit(target[:,i], source[:,i], lmbda=lmbda_cur, distance=distance[i], 
                            source_weight=source_weight, target_weight=target_weight, **kwargs)
                    if "Warning" in f.getvalue():
                        lmbda_cur *= 10
                        iter += 1
                    else:
                        adapt_flag = True
                    if iter > 5:
                        raise ValueError
                    if len(f.getvalue()) > 0:
                        print("Corrected:", f.getvalue(), flush=True)
            else:
                adapt = adaptation_num_1d()

                adapt_flag = False
                iter = 1
                while not adapt_flag:
                    f = io.StringIO()
                    with redirect_stdout(f):
                        adapt.fit(target[:,i], source[:,i], lmbda=lmbda_cur, **kwargs)
                    if "Warning" in f.getvalue():
                        lmbda_cur *= 10
                        iter += 1
                    else:
                        adapt_flag = True
                    if iter > 5:
                        raise ValueError
                    if len(f.getvalue()) > 0:
                        print("Corrected:", f.getvalue(), flush=True)
            self.adapt.append(adapt)

    def transform(self, target, repeat=1, interpolation=1, njobs=1):
        args = []
        
        if isinstance(interpolation, int) or isinstance(interpolation, float):
            interpolation = [interpolation] * (self.cate_dim + self.num_dim)

        for i in range(self.cate_dim + self.num_dim):
            args.append([self.adapt[i], target[:,i].copy(), repeat, interpolation[i]])
        
        with Pool(njobs) as p:
            target_trans = p.map(multicore_helper, args)
        
        return np.array(target_trans).T


    def reverse_transform(self, source, repeat=1, interpolation=1, njobs=1):
        args = []
        
        if isinstance(interpolation, int) or isinstance(interpolation, float):
            interpolation = [interpolation] * (self.cate_dim + self.num_dim)

        source_trans = []
        for i in range(self.cate_dim + self.num_dim):
            # source_trans.append(self.adapt[i].reverse_transform(source[:,i], repeat, interpolation[i]))
            args.append([self.adapt[i], source[:,i].copy(), repeat, interpolation[i]])
        
        with Pool(njobs) as p:
            source_trans = p.map(reverse_multicore_helper, args)
        
        return np.array(source_trans).T


class adaptation_cate_1d():
    def fit(self, target, source, distance=None, lmbda=None, source_weight=None, target_weight=None, **args):
        if source_weight is None:
            source_weight = np.ones(len(source))
        if target_weight is None:
            target_weight = np.ones(len(target))


        self.identical = False
        if len(target)==0 or len(source)==0:
            self.identical = True
        else:            
            # Get target modality
            self.target_modality = np.unique(target)
            counts = np.array([target_weight[target==self.target_modality[i]].sum() for i in range(len(self.target_modality))])
            self.target_density = counts / counts.sum()

            self.target_modality = self.target_modality[self.target_density!=0]
            self.target_density = self.target_density[self.target_density!=0]

            # Get source modality
            self.source_modality = np.unique(source)
            counts = np.array([source_weight[source==self.source_modality[i]].sum() for i in range(len(self.source_modality))])
            self.source_density = counts / counts.sum()

            self.source_modality = self.source_modality[self.source_density!=0]
            self.source_density = self.source_density[self.source_density!=0]

            # Get similarity matrix
            if distance is None:
                sim, modality = of_uni_cate(source, target)
                distance = similarity_to_dissimilarity(sim)
            else:
                modality = np.unique(np.r_[target, source])     
        
            # Compute transportation plan
            target_index = np.where(np.in1d(modality, self.target_modality))[0]
            source_index = np.where(np.in1d(modality, self.source_modality))[0]

            if lmbda is None:
                Gs = pot.emd(self.target_density.tolist(), self.source_density.tolist(), 
                            distance[target_index][:,source_index].tolist())
            else:
                Gs = pot.sinkhorn(self.target_density.tolist(), self.source_density.tolist(), 
                            distance[target_index][:,source_index].tolist(), lmbda)

            # Gs = pot.lp.emd_1d(self.target_modality, self.source_modality, 
            #     self.target_density.tolist(), self.source_density.tolist())
            
            norm_array = Gs.sum(axis=1)
            norm_array[norm_array==0] = 1
            self.ot_plan = (Gs.T / norm_array).T
            
            # Get stochastic transportation plan in a dict
            self.stochastic_trans = {}
            for m in self.target_modality:
                index = np.where(self.target_modality==m)[0][0]
                self.stochastic_trans[m] = self.ot_plan[index]


            norm_array = Gs.sum(axis=0)
            norm_array[norm_array==0] = 1
            self.ot_plan_r = (Gs / norm_array).T

            self.stochastic_trans_r = {}
            for m in self.source_modality:
                index = np.where(self.source_modality==m)[0][0]
                self.stochastic_trans_r[m] = self.ot_plan_r[index]

    def transform(self, target, repeat=1, interpolation=1, **args):
        if self.identical:
            return np.tile(target, repeat)
        # Transform
        trans_target = np.tile(target, repeat)

        if interpolation > 0.5:
            trans_target_ref = trans_target.copy()
            for m in self.target_modality:
                if np.sum(trans_target_ref==m) > 0:
                    trans_target[trans_target_ref==m] = self.source_modality[np.random.choice(
                        range(len(self.source_modality)), 
                        size=np.sum(trans_target_ref==m), 
                        p=self.stochastic_trans[m])]
        
        return trans_target

    def reverse_transform(self, source, repeat=1, interpolation=1, **args):
        if self.identical:
            return np.tile(source, repeat)
        # Transform
        trans_source = np.tile(source, repeat)

        if interpolation > 0.5:
            trans_source_ref = trans_source.copy()
            for m in self.source_modality:
                if np.sum(trans_source_ref==m) > 0:
                    trans_source[trans_source_ref==m] = self.target_modality[np.random.choice(
                        range(len(self.target_modality)), 
                        size=np.sum(trans_source_ref==m), 
                        p=self.stochastic_trans_r[m])]
        
        return trans_source


class adaptation_num_1d():
    def fit(self, target, source, limit_modality=2000, lmbda=None, source_weight=None, target_weight=None, **args):
        if source_weight is None:
            source_weight = np.ones(len(source))
        if target_weight is None:
            target_weight = np.ones(len(target))

        # self.modality = np.unique(np.r_[target, source])
        # density estimation of numerical features
        # if len(self.modality) > limit_modality:
        #     print(len(self.modality), flush=True)

        # Get target modality
        self.target_modality = np.unique(target)
        counts = np.array([target_weight[target==self.target_modality[i]].sum() for i in range(len(self.target_modality))])
        self.target_density = counts / counts.sum()

        self.target_modality = self.target_modality[self.target_density!=0]
        self.target_density = self.target_density[self.target_density!=0]

        # Get source modality
        self.source_modality = np.unique(source)
        counts = np.array([source_weight[source==self.source_modality[i]].sum() for i in range(len(self.source_modality))])
        self.source_density = counts / counts.sum()

        self.source_modality = self.source_modality[self.source_density!=0]
        self.source_density = self.source_density[self.source_density!=0]

        # self.adapt = adaptation_cate_1d()
        # self.adapt.fit(target, source, None, lmbda, source_weight, target_weight, True)
        # self.adapt.fit(target, source, distance, lmbda, source_weight, target_weight, False)

        if lmbda is None:
            # Gs = pot.emd(self.target_density.tolist(), self.source_density.tolist(), 
            #             distance.tolist())
            Gs = pot.lp.emd_1d(self.target_modality, self.source_modality, 
                self.target_density.tolist(), self.source_density.tolist())
        else:
            # # Compute the distance
            distance = pot.dist(self.target_modality.reshape(-1, 1), self.source_modality.reshape(-1, 1))

            Gs = pot.sinkhorn(self.target_density.tolist(), self.source_density.tolist(), 
                        distance.tolist(), lmbda)

        norm_array = Gs.sum(axis=1)
        norm_array[norm_array==0] = 1
        self.ot_plan = (Gs.T / norm_array).T
        
        # Get stochastic transportation plan in a dict
        self.stochastic_trans = {}
        for m in self.target_modality:
            index = np.where(self.target_modality==m)[0][0]
            self.stochastic_trans[m] = self.ot_plan[index]


        norm_array = Gs.sum(axis=0)
        norm_array[norm_array==0] = 1
        self.ot_plan_r = (Gs / norm_array).T

        self.stochastic_trans_r = {}
        for m in self.source_modality:
            index = np.where(self.source_modality==m)[0][0]
            self.stochastic_trans_r[m] = self.ot_plan_r[index]


    def transform(self, target, repeat=1, interpolation=1, **args):

        target = target.copy()
        target[target < self.target_modality[0]] = self.target_modality[0]
        
        modality_index = np.digitize(target, self.target_modality) - 1
        target = self.target_modality[modality_index]
        
        # target_trans = self.adapt.transform(target, repeat)
        # return (target_trans - np.tile(target, repeat)) * interpolation + np.tile(target, repeat)

        trans_target = np.tile(target, repeat)

        if interpolation > 0.5:
            trans_target_ref = trans_target.copy()
            for m in self.target_modality:
                if np.sum(trans_target_ref==m) > 0:
                    trans_target[trans_target_ref==m] = self.source_modality[np.random.choice(
                        range(len(self.source_modality)), 
                        size=np.sum(trans_target_ref==m), 
                        p=self.stochastic_trans[m])]
        
        return trans_target

    
    def reverse_transform(self, source, repeat=1, interpolation=1, **args):
        source = source.copy()
        source[source < self.source_modality[0]] = self.source_modality[0]
        
        modality_index = np.digitize(source, self.source_modality) - 1
        source = self.source_modality[modality_index]
        
        # source_trans = self.adapt.transform(source, repeat)
        # return (source_trans - np.tile(source, repeat)) * interpolation + np.tile(source, repeat)

        trans_source = np.tile(source, repeat)

        if interpolation > 0.5:
            trans_source_ref = trans_source.copy()
            for m in self.source_modality:
                if np.sum(trans_source_ref==m) > 0:
                    trans_source[trans_source_ref==m] = self.target_modality[np.random.choice(
                        range(len(self.target_modality)), 
                        size=np.sum(trans_source_ref==m), 
                        p=self.stochastic_trans_r[m])]
        
        return trans_source



class adaptation_identity():
    def fit(self, **args):
        pass
    
    def transform(self, target, repeat=1, interpolation=None, **args):
        return np.tile(target, repeat)



class Discretize():
    
    def fit(self, num_data, limit_modality=1000):
        self.discrete_val = []
        for i in range(num_data.shape[-1]):
            modality = np.unique(num_data[:, i])
            if len(modality) > limit_modality:
                num_sort = np.sort(num_data[:, i])
                step_n = limit_modality
                step = int(len(num_sort) / step_n)
                modality_simple = np.unique(num_sort[::step])
                while len(modality_simple) < int(limit_modality/2):
                    step_n = step_n * 2
                    step = max(int(len(num_sort) / step_n), 1)
                    modality_simple = np.unique(num_sort[::step])

                self.discrete_val.append(np.unique(np.r_[[modality[0]], modality_simple, [modality[-1]]]))
            else:
                self.discrete_val.append(modality)
            # print(len(self.discrete_val[-1]))
    
    def transform(self, num_data):
        num_data = num_data.copy()
        for i in range(len(self.discrete_val)):
            num_data[num_data[:, i] < self.discrete_val[i][0], i] = self.discrete_val[i][0]
            
            ind = np.digitize(num_data[:, i], self.discrete_val[i]) - 1
            num_data[:, i] = self.discrete_val[i][ind]
        
        return num_data
