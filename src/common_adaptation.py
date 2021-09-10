import numpy as np
import ot as pot

from sklearn.metrics.pairwise import cosine_similarity

from metric import of_uni_cate, similarity_to_dissimilarity

class adaptation_cate_1d():
    def fit(self, target, source, distance=None, lmbda=None, **args):
        self.identical = False
        if len(target)==0 or len(source)==0:
            self.identical = True
        else:
            # Get similarity matrix
            if distance is None:
                sim, modality = of_uni_cate(source, target)
                distance = similarity_to_dissimilarity(sim)
            else:
                modality = np.unique(np.r_[target, source])
            
            # Get target modality
            self.target_modality, counts = np.unique(target, return_counts=True)
            self.target_density = counts / counts.sum()

            # Get source modality
            self.source_modality, counts = np.unique(source, return_counts=True)
            self.source_density = counts / counts.sum()
            
            # Compute transportation plan
            target_index = np.where(np.in1d(modality, self.target_modality))[0]
            source_index = np.where(np.in1d(modality, self.source_modality))[0]

            if lmbda is None:
                Gs = pot.emd(self.target_density.tolist(), self.source_density.tolist(), 
                            distance[target_index][:,source_index].tolist())
            else:
                Gs = pot.sinkhorn(self.target_density.tolist(), self.source_density.tolist(), 
                            distance[target_index][:,source_index].tolist(), lmbda)
            
            norm_array = Gs.sum(axis=1)
            norm_array[norm_array==0] = 1
            self.ot_plan = (Gs.T / norm_array).T
            
            # Get stochastic transportation plan in a dict
            self.stochastic_trans = {}
            for m in self.target_modality:
                index = np.where(self.target_modality==m)[0][0]
                self.stochastic_trans[m] = self.ot_plan[index]

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
