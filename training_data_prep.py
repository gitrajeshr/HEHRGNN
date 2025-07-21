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
        arities = torch.count_nonzero(pos_batch[:,1:self.config.max_arity+1], dim=1)

        pos_batch = torch.cat((pos_batch,torch.zeros(pos_batch.size(0),1)),1)
        batch = np.concatenate([self.neg_each(np.repeat([c], self.config.neg_ratio * arities[i] + 1, axis=0), arities[i], self.config.neg_ratio,i) for i, c in enumerate(pos_batch.numpy())], axis=0)
        batch = torch.from_numpy(batch).int().to(self.config.device)
        if(torch.count_nonzero(batch[:,-1]) >512):
            print(f"$$$$$$$$$Hey the positive indices more than 512")
            label = batch[:,-1]
            pos_indices = torch.nonzero(label).squeeze()
        
        return batch
    

    def neg_each(self, arr, arity, nr,i):
        arr[0,-1] = 1
        for a in range(arity):
            arr[a* nr + 1:(a + 1) * nr + 1, a + 1] = np.random.randint(low=1, high=self.num_ent, size=nr)
        #print(f"@@@@n neg samples={arr}")
        return arr
    
    def pos_neg_set_predictions_in_row(self,labels, predictions):
        max_length = 1 + self.config.max_arity * self.config.neg_ratio
        positive_indices = torch.nonzero(labels).squeeze()
        pos_neg_set_size = torch.ones_like(positive_indices)
        print(f"Labels = {labels.shape} Positive Indices {positive_indices.shape} predictions shape{predictions.shape}")
        seq = []
        for ind, val in enumerate(positive_indices):
            if(ind == len(positive_indices)-1):
                pos_neg_set = predictions[val:]
                padded_pos_neg_set = self.padd(pos_neg_set, max_length)                 
            else:
                pos_neg_set = predictions[val:positive_indices[ind + 1]]
                padded_pos_neg_set = self.padd(pos_neg_set, max_length)
            pos_neg_set_size[ind] = pos_neg_set.size(0)
            seq.append(padded_pos_neg_set)
            #print(f"POs neg row = {pos_neg_set.shape} {pos_neg_set}")
        pos_neg_predictions = torch.stack(seq)
        targets = torch.zeros_like(pos_neg_predictions)
        targets[:,0] = 1
        return pos_neg_predictions, targets,pos_neg_set_size
    
    def padd(self, a, max_length):
        b = F.pad(a, (0,max_length - len(a)), 'constant', -math.inf)
        return b


    