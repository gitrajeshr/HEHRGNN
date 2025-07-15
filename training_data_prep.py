import torch
import numpy as np
from torch.nn import functional as F
import math


class TrainingDataPrep():
    def __init__(self,dataset,config):
        self.config = config
        self.num_ent = dataset.num_entities

    def mark_arities(self,batch):
        arities = torch.count_nonzero(batch[:,1:self.config.max_arity+1], dim=1)
        #print(f"Arities = {arities}")
        np_pres_bits = np.zeros((len(batch),self.config.max_arity))
        np_abs_bits = np.ones((len(batch),self.config.max_arity))
        for i in range(len(batch)):
            np_pres_bits[i][0:arities[i]] = 1
            np_abs_bits[i][0:arities[i]] = 0

        pres_bits = torch.from_numpy(np_pres_bits).int().to(self.config.device)

        abs_bits = torch.from_numpy(np_abs_bits).int().to(self.config.device)

        return pres_bits,abs_bits

    def add_neg_samples(self, pos_batch):
        #print(f"++++++ve Samples {pos_batch}")
        arities = torch.count_nonzero(pos_batch[:,1:self.config.max_arity+1], dim=1)

        #pos_batch[:,-1] = arities
        torch.cat((pos_batch,torch.zeros(pos_batch.size(0),1)),1)
        neg_batch = np.concatenate([self.neg_each(np.repeat([c], self.config.neg_ratio * arities[i] + 1, axis=0), arities[i], self.config.neg_ratio) for i, c in enumerate(pos_batch.numpy())], axis=0)

        
        return torch.from_numpy(neg_batch).int().to(self.config.device)
    

    def neg_each(self, arr, arity, nr):
        arr[0,-1] = 1
        for a in range(arity):
            arr[a* nr + 1:(a + 1) * nr + 1, a + 1] = np.random.randint(low=1, high=self.num_ent, size=nr)

        return arr
    
    def pos_neg_set_predictions_in_row(self,labels, predictions):
        max_length = self.config.max_arity * self.config.neg_ratio
        positive_indices = torch.nonzero(labels).squeeze()
        #print(f"Maxlength = {max_length} Positive Indices {positive_indices} predictions shape{predictions.shape} labels = {labels}")
        seq = []
        for ind, val in enumerate(positive_indices):
            if(ind == len(positive_indices)-1):
                pos_neg_row = self.padd(predictions[val:], max_length)                 
            else:
                pos_neg_row = self.padd(predictions[val:positive_indices[ind + 1]], max_length)
            
            seq.append(pos_neg_row)
            #print(f"POs neg row = {pos_neg_row.shape} {pos_neg_row}")
        pos_neg_set = torch.stack(seq)
        targets = torch.zeros_like(pos_neg_set)
        targets[:,0] = 1
        return pos_neg_set, targets
    
    def padd(self, a, max_length):
        b = F.pad(a, (0,max_length - len(a)), 'constant', -math.inf)
        return b


    