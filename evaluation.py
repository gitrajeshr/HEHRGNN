from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import torch
import numpy as np
from training_data_prep import TrainingDataPrep



class EvaluatorClass():
    def __init__(self,model,dataset,config):
          self.dataset = dataset
          self.model = model
          self.config = config
          pass       
    def compute_rank_metrics(self, predictions, targets, pos_neg_set_size,metrics):

        b_range = torch.arange(predictions.size()[0], device=self.config.device)

        exclude_ents = []
        
        """ ents_to_ignore = label.clone()
        ents_to_ignore[b_range, targets] = 0        
        ents_to_ignore[:, exclude_ents] = 1     
        predictions[ents_to_ignore.bool()] = -1000000 """
        #Target is always 0 because our +ve tuple is the first one in each row
        ranks = 1 + torch.argsort(torch.argsort(predictions, dim=1, descending=True), dim=1, descending=False)[b_range, 0]
        #print(f"#####EValuation Ranks={ranks} \n Predictions = {predictions[0],predictions[1]}")
        ranks = ranks.float()
        metrics['count'] = torch.numel(ranks) + metrics.get('count', 0.0)
        metrics['mr'] = torch.sum(ranks).item() + metrics.get('mr', 0.0)
        metrics['mrr'] = torch.sum(1.0 / ranks).item() + metrics.get('mrr', 0.0)
        for k in [1, 3, 5, 10]:
            metrics['hits_at {}'.format(k )] = torch.numel(ranks[ranks <= (k )]) + metrics.get(
                'hits_at {}'.format(k ), 0.0)
            if k == 10:
                print(f"Hits@10={torch.numel(ranks[ranks <= (k )])} Total ranks={torch.numel(ranks)}")
        return metrics

   
    
    def evaluate(self,epoch_num):
        model = self.model
        config = self.config
        dataset = self.dataset
        model.eval()
        dataloader = DataLoader(dataset.data["test"],self.config.batch_size)
        iter_dataset = iter(dataloader)
        accumulated_metrics={}
        summary_metrics={}
        batch_counter=1
        data_prep = TrainingDataPrep(dataset,config)
        for pos_batch in tqdm(iter_dataset):
            batch_counter+=1
            
            batch = data_prep.add_neg_samples(pos_batch).to(config.device,dtype=torch.long)

            pres_bits,abs_bits = data_prep.mark_arities(batch)


            rel, ent1, ent2, ent3, ent4, ent5, ent6,labels = batch[:,0], batch[:,1],batch[:,2],batch[:,3],batch[:,4],batch[:,5],batch[:,6],batch[:,-1]
            #print(f">>>>>>>.batch={batch[0:2,:config.max_arity+1]}..pres_bits = {pres_bits} abs_bits={abs_bits}")

            predictions = model(dataset.graph_representation,rel,ent1,ent2,ent3,ent4,ent5,ent6,pres_bits,abs_bits)
            print(f">>>>>>>predictions ={predictions.shape} total batch size={batch.shape}")


            predictions,targets,pos_neg_set_size = data_prep.pos_neg_set_predictions_in_row(labels,predictions)
            #print(f">>>>>>>..EValuation .Target{targets.shape} targets={targets}")

            #print(f">>>>>>>POs set Predictions {predictions.shape}..batch = {batch[0],batch[1]}")
            self.compute_rank_metrics(predictions,targets,pos_neg_set_size,accumulated_metrics)
        
        for k, v in accumulated_metrics.items():
            print(f" In summary metrics k={k} v={v} dataset len = {float(len(dataloader.dataset))}")
            summary_metrics[k] = v / float(len(dataloader.dataset)) if k != 'count' else v
        return summary_metrics