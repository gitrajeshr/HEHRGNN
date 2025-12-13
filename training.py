from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from evaluation import EvaluatorClass
from training_data_prep import TrainingDataPrep
import os
from datetime import datetime


class Training():
    def __init__(self,model,config):
        self.model= model
        self.config = config
        self.loss_layer, self.optimizer = self.set_loss_n_optimizer()
        self.best_model = None
        self.output_dir = "./models"


    def set_loss_n_optimizer(self):
        config = self.config
        model = self.model
        if config.loss == 'CEL':
            loss_layer = torch.nn.CrossEntropyLoss()
        elif config.loss == 'BCEL':
            loss_layer = torch.nn.BCELoss()
        else:
            print("Unexpected loss")
            raise NotImplementedError
        
        if config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)

        else:
            print("Unexpected optimizer")
            raise NotImplementedError
        
        return loss_layer, optimizer
   

    def save_if_best_model(self,model,metrics,best_metrics, epch,log_filename):
        
            # This is the best model we have so far if
            # no "best_model" exists yes, or if this MRR is better than what we had before
            is_best_model = (best_metrics.get("mrr") is None ) or (metrics["mrr"] > best_metrics["mrr"])
            if is_best_model:
                self.best_model = self.model
                # Update the best_mrr value
                best_metrics["mrr"] = metrics["mrr"]
                for k in [1, 3, 5, 10]:
                    best_metrics['hits_at {}'.format(k )] = metrics['hits_at {}'.format(k )]
                 
                
            # Save the model at checkpoint
            model_details_str = log_filename.rsplit("/", 1)[-1][:-len("_training_log.txt")]
            if is_best_model:
                saved_model_name = model_details_str+'_best_model.chkpnt' 
                torch.save(model.state_dict(), os.path.join(self.config.output_dir, saved_model_name))
                print(f"######## Saving the BEST MODEL in path {saved_model_name}")
            if (epch % 10 == 0):
                saved_model_name = model_details_str+'_epch_{}_model.chkpnt'.format(epch)
                opt_name = model_details_str +'_epch_{}_opt.chkpnt'.format(epch) 
                print("######## Saving the model {}".format(saved_model_name))

                torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, saved_model_name))
                torch.save(self.optimizer.state_dict(), os.path.join(self.config.output_dir, opt_name))
     
    


    def training_loop(self,dataset):
    #Do the training loop
    #convert  dataset to iterable
        config = self.config
        model = self.model
        train_data = dataset.data["train"]
        #print(f">>>>>>>>original dataset = {train_data}")
        dataloader = DataLoader(train_data,config.batch_size)    
        #torch.set_default_device(config.device)
        if(config.inductive ==1):
            ind_trans= "inductive"
        else:
            ind_trans= "transductive"

        log_file_name = os.path.join("results",f"{(datetime.now()).strftime('%Y%m%d_%H%M%S')}_{dataset.name}_{config.decoder}_{config.loss}_embdim-{config.emb_dim}_epochs-{config.epochs}_numgnn-{config.num_gnn_layers}_BN-{config.use_bn}_induct-{config.inductive}_edgeEmb-{config.edge_embed}_shalEmb-{config.shallow_embed}_training_log.txt")

        #log_file = open(log_file_name, "w")
        opt = self.optimizer
        train_loss = []
        data_prep = TrainingDataPrep(dataset,config)
        evaluator = EvaluatorClass(model,dataset,config)
        best_metrics={}
        for epch in range(config.epochs):
            model.train()
            batch_counter=0
            batch_losses = []
            iter_dataset = iter(dataloader)
            for pos_batch in tqdm(iter_dataset):
                log_file = open(log_file_name, "a")
                print(f"Epoch No.{epch} Batch No.{batch_counter} ")
                batch_counter+=1
                opt.zero_grad()
                #pos_batch = pos_batch.to(config.device,dtype=torch.long)
                #add_neg_samples second param is train_or_eval which is 1 for eval and 0 for train
                batch = data_prep.add_neg_samples(pos_batch,0).to(config.device,dtype=torch.long)

                pres_bits,abs_bits = data_prep.mark_arities(batch)


                rel, ent1, ent2, ent3, ent4, ent5, ent6,labels = batch[:,0], batch[:,1],batch[:,2],batch[:,3],batch[:,4],batch[:,5],batch[:,6],batch[:,-1]
                print(f">>>>>>>Pos_batch size = {pos_batch.shape} labels shape={labels.shape} positives={torch.numel(labels[labels!=0])}")

                predictions = model(dataset.graph_representation,rel,ent1,ent2,ent3,ent4,ent5,ent6,pres_bits,abs_bits)
                print(f"IN training predictions shape={predictions.shape}")
                #predictions is the score for all samples in the batch, +ve as well the corresponding -ves. 
                #Predictions is of shape(bs,1)
                #We'll transform the shape into (pos_bs, 1+num_neg_samples) and then apply BCEloss
                predictions,targets,pos_neg_set_size = data_prep.pos_neg_set_predictions_in_row(labels,predictions)
                #predictions is raw logit values for all the entities i.e the likelihood of each entity 
                # to be the predicted/masked entity
                if(config.loss == 'CEL'):
                    #pytorch CEL expects raw logit values. So no operations required
                    loss = self.loss_layer(predictions, targets)
                else:
                    #pytorch BCEL expects probability values for the +ve class. Hence sigmoid is to be applied
                    predictions = torch.sigmoid(predictions)
                    print(f"^^^^^Before Loss predictions shape={predictions.shape} targets={targets.shape}")
                    loss = self.loss_layer(predictions, targets)
                batch_losses.append(loss.item())
                print(f">>>>>>>Batch Loss...{loss}")
                log_file.write(f"Epoch {epch} Batch {batch_counter} Loss: {loss} \n")
                loss.backward()
                #with amp.scale_loss(loss, opt) as scaled_loss:
                            #     scaled_loss.backward()
                #if grad_clipping:
                #    torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), 1.0)
                opt.step()
                log_file.close()

            # Log this stuff
            epoch_loss = np.mean(batch_losses)
            print(f"[Epoch: {epch} ] Loss: {epoch_loss} Batch Losses={batch_losses}")
            # train_acc.append(np.mean(per_epoch_tr_acc))
            train_loss.append(epoch_loss)

            # Evaluate the model every 100th iteration or if it is the last iteration
            if (epch % 1 == 0) or (epch == config.epochs):
                log_file = open(log_file_name, "a")
                # with torch.no_grad()
                # model.eval() # both these setting are done in Eval. No need to do it here
                metrics = evaluator.evaluate()
                log_file.write(f"EPoch {epch} Evaluation Metrics {metrics} \n")
                self.save_if_best_model(model,metrics,best_metrics,epch,log_file_name)
                log_file.close()

    
