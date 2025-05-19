from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np




def set_loss_n_optimizer(model,config):
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
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rat)
    elif config.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.learning_rate)

    else:
        print("Unexpected optimizer")
        raise NotImplementedError
    
    return loss_layer, optimizer

def training_loop(model,dataset,loss_layer,opt,config):
#Do the training loop
#convert  dataset to iterable
    dataloader = DataLoader(dataset.data["train"],batch_size=2)
    
    #torch.set_default_device(config.device)
    train_loss = []
    for epch in range(config.epochs):
        model.train()
        batch_counter=0
        batch_losses = []
        iter_dataset = iter(dataloader)
        for batch in tqdm(iter_dataset):
            print(f"Epoch No.{epch} Batch No.{batch_counter}  batch={batch}")
            batch_counter+=1
            opt.zero_grad()
            batch = batch.to(config.device,dtype=torch.long)

            rel, ent1, targets = batch[:,0], batch[:,1],batch[:,2]

            predictions = model(dataset.graph_representation,rel,ent1)

            loss = loss_layer(predictions, targets)
            batch_losses.append(loss.item())
            #print(f">>>>>>>Loss{loss.shape}...{loss}")
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
   