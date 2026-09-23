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
        # position j is present iff j < arity of that row
        pres_bits = (torch.arange(self.config.max_arity, device=batch.device).unsqueeze(0) < arities.unsqueeze(1)).int().to(self.config.device)
        abs_bits = 1 - pres_bits
        return pres_bits,abs_bits

    def add_neg_samples(self, pos_batch,train_or_eval):
        arities = torch.count_nonzero(pos_batch[:,1:self.config.max_arity+1], dim=1)
        
        if train_or_eval == 1: #1 is eval, 0 is train
            neg_ratio = self.num_ent // 100
            #neg_ratio = self.num_ent
        else:
            neg_ratio = self.config.neg_ratio


        device = self.config.device
        pos_batch = pos_batch.to(device)
        arities = arities.to(device)
        # Each tuple i expands to a block of 1 + neg_ratio*arity_i rows: the positive first, then
        # neg_ratio corrupted rows for each of its arity_i entity positions.
        block_sizes = neg_ratio * arities + 1
        row_to_tuple = torch.repeat_interleave(torch.arange(pos_batch.size(0), device=device), block_sizes)
        block_starts = torch.cumsum(block_sizes, dim=0) - block_sizes
        offset = torch.arange(row_to_tuple.numel(), device=device) - block_starts[row_to_tuple]

        batch = torch.cat((pos_batch[row_to_tuple], (offset == 0).unsqueeze(1).to(pos_batch.dtype)), dim=1)
        neg_rows = torch.nonzero(offset > 0).squeeze(1)
        # offset 1..nr corrupts entity column 1, nr+1..2nr column 2, ...
        neg_cols = (offset[neg_rows] - 1) // neg_ratio + 1
        batch[neg_rows, neg_cols] = torch.randint(1, self.num_ent, (neg_rows.numel(),), device=device, dtype=batch.dtype)
        batch = batch.int()
        if(torch.count_nonzero(batch[:,-1]) >512):
            print(f"$$$$$$$$$Hey the positive indices more than 512")
            label = batch[:,-1]
            pos_indices = torch.nonzero(label).squeeze()
        
        return batch
    

    def pos_neg_set_predictions_in_row(self,labels, predictions):
        max_length = 1 + self.config.max_arity * self.config.neg_ratio
        positive_indices = torch.nonzero(labels).flatten()
        # each set runs from a positive up to the next positive (or the end of the batch)
        ends = torch.cat((positive_indices[1:], positive_indices.new_tensor([predictions.size(0)])))
        pos_neg_set_size = ends - positive_indices
        print(f"Labels = {labels.shape} Positive Indices {positive_indices.shape} predictions shape{predictions.shape}")
        set_id = torch.repeat_interleave(torch.arange(positive_indices.numel(), device=predictions.device), pos_neg_set_size)
        col = torch.arange(predictions.size(0), device=predictions.device) - positive_indices[set_id]
        # pad short sets with -inf; sets longer than max_length are cut, as F.pad did
        keep = col < max_length
        pos_neg_predictions = torch.full((positive_indices.numel(), max_length), -math.inf, dtype=predictions.dtype, device=predictions.device)
        pos_neg_predictions[set_id[keep], col[keep]] = predictions[keep]
        targets = torch.zeros_like(pos_neg_predictions)
        targets[:,0] = 1
        return pos_neg_predictions, targets,pos_neg_set_size
    
    def padd(self, a, max_length):
        b = F.pad(a, (0,max_length - len(a)), 'constant', -math.inf)
        return b


    
