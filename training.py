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
   

    def save_if_best_model(self,model,config,metrics,best_metrics, epch,log_filename):
        
            # This is the best model we have so far if
            # no "best_model" exists yes, or if this MRR is better than what we had before
            if config.task == "link_prediction":

                is_best_model = (best_metrics.get("mrr") is None ) or (metrics["mrr"] > best_metrics["mrr"])
                if is_best_model:
                    self.best_model = self.model
                    # Update the best_mrr value
                    best_metrics["mrr"] = metrics["mrr"]
                    for k in [1, 3, 5, 10]:
                        best_metrics['hits_at {}'.format(k )] = metrics['hits_at {}'.format(k )]
            else:
                is_best_model = (best_metrics.get("accuracy") is None ) or (metrics["accuracy"] > best_metrics["accuracy"])
                print(f"Best accuracy so far={best_metrics.get('accuracy')} Current accuracy={metrics['accuracy']}")
                if is_best_model:
                    print(f"New best accuracy: {metrics['accuracy']}")
                    self.best_model = self.model
                    # Update the best_mrr value
                    best_metrics["accuracy"] = metrics["accuracy"]
                
            # Save the model at checkpoint
            model_details_str = log_filename.rsplit("/", 1)[-1][:-len("_training_log.txt")]
            if is_best_model:
                saved_model_name = model_details_str+'_best_model.chkpnt' 
                torch.save(model.state_dict(), os.path.join(self.config.output_dir, saved_model_name))
                self.saved_best_model = saved_model_name

                print(f"######## Saving the BEST MODEL in path {saved_model_name}")
            if (epch % 10 == 0):
                saved_model_name = model_details_str+'_epch_{}_model.chkpnt'.format(epch)
                opt_name = model_details_str +'_epch_{}_opt.chkpnt'.format(epch) 
                print("######## Saving the model {}".format(saved_model_name))

                torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, saved_model_name))
                torch.save(self.optimizer.state_dict(), os.path.join(self.config.output_dir, opt_name))
     
    

    def training_loop(self,dataset):
        if self.config.task == "link_prediction":
            self.link_prediction_training_loop(dataset)
        elif self.config.task == "node_classification":
            self.node_classification_training_loop(dataset)
        else:
            print("Unexpected task")
            raise NotImplementedError
        
    def link_prediction_training_loop(self,dataset):
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
                task = "link_prediction"
                input_for_pred = {}
                input_for_pred["r_id"] = rel
                input_for_pred["e1_id"] = ent1
                input_for_pred["e2_id"] = ent2
                input_for_pred["e3_id"] = ent3
                input_for_pred["e4_id"] = ent4
                input_for_pred["e5_id"] = ent5
                input_for_pred["e6_id"] = ent6
                input_for_pred["pres_bits"] = pres_bits
                input_for_pred["abs_bits"] = abs_bits
                predictions = model(dataset.graph_representation,task,input_for_pred)
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
                ##!!!We cannot directly use the embeddings generated during the last training
                # iteration and instead need to do a fresh forward pass for evaluation because 
                # the weights have been updated in the last training iteration 
                metrics = evaluator.evaluate(split="val")
                log_file.write(f"EPoch {epch} Evaluation Metrics {metrics} \n")
                self.save_if_best_model(model,config,metrics,best_metrics,epch,log_file_name)
                log_file.close()

    def node_classification_training_loop(self,dataset):
    
        config = self.config
        model = self.model
        # training_nodes_mask = dataset.data["nodes"]
        # #print(f">>>>>>>>original dataset = {train_data}")
        # dataloader = DataLoader(train_data,config.batch_size)  
        log_file_name = os.path.join("results",f"{(datetime.now()).strftime('%Y%m%d_%H%M%S')}_{dataset.name}_{config.decoder}_{config.loss}_embdim-{config.emb_dim}_epochs-{config.epochs}_numgnn-{config.num_gnn_layers}_BN-{config.use_bn}_induct-{config.inductive}_edgeEmb-{config.edge_embed}_shalEmb-{config.shallow_embed}_training_log.txt")
        opt = self.optimizer
        train_loss = []
        best_metrics={}
        evaluator = EvaluatorClass(model,dataset,config)

        for epch in range(config.epochs):
            model.train()
            batch_counter=0
            batch_losses = []
            opt.zero_grad()
            log_file = open(log_file_name, "a")
            # The model processes EVERY node and edge in the graph
            task = "node_classification"
            input_for_pred = {}
            input_for_pred["train_mask"] = dataset.data["train_mask"]
            input_for_pred["val_mask"] = dataset.data["val_mask"]
            input_for_pred["test_mask"] = dataset.data["test_mask"]
            input_for_pred["labels"] = dataset.data["labels"]
            
            predictions = model(dataset.graph_representation,task,input_for_pred)
            print(f"0000000IN loss predictions shape={predictions.shape} train_mask shape={dataset.data['train_mask'].shape} ")
            #print(f"IN loss predictions shape={predictions[dataset.data['train_mask']].shape} labels shape={dataset.data['labels'].shape} label shape={dataset.data['labels'][dataset.data['train_mask']].shape}")

            # CRITICAL: Loss is ONLY calculated using the training mask labels
            loss = self.loss_layer(predictions[dataset.data["train_mask"]], dataset.data["labels"][dataset.data["train_mask"]])
            batch_losses.append(loss.item())
            print(f">>>>>>>Batch Loss...{loss}")
            log_file.write(f"Epoch {epch} Batch {batch_counter} Loss: {loss} \n")
            loss.backward()
            self.optimizer.step()

            # Log this stuff
            epoch_loss = np.mean(batch_losses)
            print(f"[Epoch: {epch} ] Loss: {epoch_loss} Batch Losses={batch_losses}")
            # train_acc.append(np.mean(per_epoch_tr_acc))
            train_loss.append(epoch_loss)

            #Evaluate the model every 100th iteration or if it is the last iteration
            if (epch % 1 == 0) or (epch == config.epochs):
                #log_file = open(log_file_name, "a")
                # with torch.no_grad()
                # model.eval() # both these setting are done in Eval. No need to do it here
                metrics = evaluator.evaluate(split="val")
                log_file.write(f"EPoch {epch} Evaluation Metrics {metrics} \n")
                self.save_if_best_model(model,config,metrics,best_metrics,epch,log_file_name)
            log_file.close()
        #load best chkpnt and do final evaluation on test set
        #final test evaluation with the best model
        self.model.load_state_dict(torch.load(os.path.join(self.config.output_dir, self.saved_best_model)))


        log_file = open(log_file_name, "a")
        metrics = evaluator.evaluate(split="test")
        log_file.write(f"Hyper parameters: {config} \n")
        log_file.write(f"Final test Evaluation Metrics {metrics} \n")
        log_file.close()


    