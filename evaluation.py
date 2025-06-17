from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import torch



class EvaluatorClass():
    def __init__(self,model,dataset,config):
          self.dataset = dataset
          self.model = model
          self.config = config
          pass       
    def compute_rank_metrics(self, predictions, targets, metrics):

        b_range = torch.arange(predictions.size()[0], device=self.config.device)

        exclude_ents = []
        
        """ ents_to_ignore = label.clone()
        ents_to_ignore[b_range, targets] = 0        
        ents_to_ignore[:, exclude_ents] = 1     
        predictions[ents_to_ignore.bool()] = -1000000 """
        
        ranks = 1 + torch.argsort(torch.argsort(predictions, dim=1, descending=True), dim=1, descending=False)[b_range, targets]
        print(f"Ranks computed = {ranks}")
        ranks = ranks.float()
        metrics['count'] = torch.numel(ranks) + metrics.get('count', 0.0)
        metrics['mr'] = torch.sum(ranks).item() + metrics.get('mr', 0.0)
        metrics['mrr'] = torch.sum(1.0 / ranks).item() + metrics.get('mrr', 0.0)
        for k in [1, 3, 5, 10]:
            metrics['hits_at {}'.format(k )] = torch.numel(ranks[ranks <= (k )]) + metrics.get(
                'hits_at {}'.format(k ), 0.0)
        return metrics

    
    def evaluate(self,epoch_num):
        model = self.model
        model.eval()
        dataloader = DataLoader(self.dataset.data["test"],self.config.batch_size)
        iter_dataset = iter(dataloader)
        accumulated_metrics={}
        summary_metrics={}
        batch_counter=1
        for batch in tqdm(iter_dataset):
            print(f"Test Batch No.{batch_counter}")
            batch_counter+=1
            batch = batch.to(self.config.device,dtype=torch.long)

            rel, ent1, targets = batch[:,0], batch[:,1],batch[:,2]

            predictions = model(self.dataset.graph_representation,rel,ent1)
            print(f">>>>>>>Predictions {predictions.shape}..")
            self.compute_rank_metrics(predictions,targets,accumulated_metrics)
        
        for k, v in accumulated_metrics.items():
            summary_metrics[k] = v / float(len(dataloader.dataset)) if k != 'count' else v
        return summary_metrics