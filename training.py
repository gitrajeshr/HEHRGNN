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
    def save_model(self, itr=0, test_or_valid='test', is_best_model=False):
            """
            Save the model state to the output folder.
            If is_best_model is True, then save the model also as best_model.chkpnt
            """
            if is_best_model:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.chkpnt'))
                print(f"######## Saving the BEST MODEL in path {os.path.join(self.output_dir, 'best_model.chkpnt')}")

            model_name = 'model_{}itr.chkpnt'.format(itr)
            opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(self.model_name)
            #measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)
            print("######## Saving the model {}".format(os.path.join(self.output_dir, model_name)))

            torch.save(self.model.state_dict(), os.path.join(self.output_dir, model_name))
            torch.save(self.optimizer.state_dict(), os.path.join(self.output_dir, opt_name))
            """ if self.measure is not None:
                measure_dict = vars(self.measure)
                # If a best model exists
                if self.best_model:
                    measure_dict["best_iteration"] = self.best_model.best_itr.cpu().item()
                    measure_dict["best_mrr"] = self.best_model.best_mrr.cpu().item()
                with open(os.path.join(self.output_dir, measure_name), 'w') as f:
                        json.dump(measure_dict, f, indent=4, sort_keys=True)
            # Note that measure_by_arity is only computed at test time (not validation)
            if (self.test_by_arity) and (self.measure_by_arity is not None):
                H = {}
                measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid, itr) if itr else '{}.json'.format(self.model_name)
                for key in self.measure_by_arity:
                    H[key] = vars(self.measure_by_arity[key])
                with open(os.path.join(self.output_dir, measure_by_arity_name), 'w') as f:
                        json.dump(H, f, indent=4, sort_keys=True) """

    def save_if_best_model(self,epch,dataset):
        
            # This is the best model we have so far if
            # no "best_model" exists yes, or if this MRR is better than what we had before
            is_best_model = (self.best_model is None) or (mrr > self.best_model.best_mrr)
            if is_best_model:
                self.best_model = self.model
                # Update the best_mrr value
                self.best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                self.best_model.best_itr.data = torch.from_numpy(np.array([it]))
            # Save the model at checkpoint
            self.save_model(epch, "valid", is_best_model=is_best_model)
            print("This validation Over") 
     
    


    def training_loop(self,dataset):
    #Do the training loop
    #convert  dataset to iterable
        config = self.config
        model = self.model
        train_data = dataset.data["train"]
        #print(f">>>>>>>>original dataset = {train_data}")
        dataloader = DataLoader(train_data,config.batch_size)    
        #torch.set_default_device(config.device)
        log_file_name = os.path.join("results",f"{(datetime.now()).strftime('%Y%m%d_%H%M%S')}_{dataset.name}_{config.decoder}_{config.loss}_embdim-{config.emb_dim}_epochs-{config.epochs}_training_log.txt")
        log_file = open(log_file_name, "w")
        opt = self.optimizer
        train_loss = []
        data_prep = TrainingDataPrep(dataset,config)
        evaluator = EvaluatorClass(model,dataset,config)
        for epch in range(config.epochs):
            model.train()
            batch_counter=0
            batch_losses = []
            iter_dataset = iter(dataloader)
            for pos_batch in tqdm(iter_dataset):
                print(f"Epoch No.{epch} Batch No.{batch_counter} ")
                batch_counter+=1
                opt.zero_grad()
                #pos_batch = pos_batch.to(config.device,dtype=torch.long)

                batch = data_prep.add_neg_samples(pos_batch).to(config.device,dtype=torch.long)

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

            # Log this stuff
            epoch_loss = np.mean(batch_losses)
            print(f"[Epoch: {epch} ] Loss: {epoch_loss} Batch Losses={batch_losses}")
            # train_acc.append(np.mean(per_epoch_tr_acc))
            train_loss.append(epoch_loss)

            # Evaluate the model every 100th iteration or if it is the last iteration
            if (epch % 1 == 0) or (epch == config.epochs):
                model.eval()
                with torch.no_grad():
                    metrics = evaluator.evaluate(epch)
                    print(f" EValuation results {metrics}")
                    log_file.write(f"EPoch {epch} Evaluation Metrics {metrics} \n")
        log_file.close()

    