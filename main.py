import argparse
import torch


from graph_encoder import GraphEncoder
from data_input import InputDataManager

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HEHR")
    parser.add_argument('-dataset', type=str, default="wd50k_new_format")
    parser.add_argument('-max_arity', type=str, default=6) 
    parser.add_argument('-max_q_pairs', type=str, default=20)   
     
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-optimizer', type=str, default="sgd")

    config = parser.parse_args()
    dataset = InputDataManager(config)
    print(f"Dataset loaded ...{dir(dataset)}...\ncontains {dataset.graph_representation.keys()} .....\nGraph Repsn{len(dataset.ent2id)}")

    graph_encoder = GraphEncoder(dataset,config).to(config.device)

    #Now we have the embeddings for entities as well as relations. How do we evaluate?
    #The encoder decoder should be part of the GraphEncoder in order to optimize
    #  the embedding based on the reconstruction loss?
    #Add the loss functions
    #What is the default optimizer used by nn.Modules? If we don't specify any optimizer
    #for GraphEncoder, does it use a default one? Even for GNNlayer?
    #Where do we put the optimizer, it should be 
    loss_layer, opt = set_loss_n_optimizer(graph_encoder,config)
    


    #Do the training loop
    
    opt.zero_grad()

    graph_encoder(dataset.graph_representation)

    loss = loss_layer(predictions, targets)
    per_epoch_loss.append(loss.item())
    losses += loss.item()
    #print(f">>>>>>>Loss{loss.shape}...{loss}")
    loss.backward()
    #with amp.scale_loss(loss, opt) as scaled_loss:
                #     scaled_loss.backward()
    #if grad_clipping:
    #    torch.nn.utils.clip_grad_norm_(graph_encoder.parameters(), 1.0)
    opt.step()
    
    print(">>>>>>>>>Graph Encoder Model params: ",sum([param.nelement() for param in graph_encoder.parameters()]))
    for name, param in graph_encoder.named_parameters():
        if param.requires_grad:
            print(f"name={name}, param.data = {param.shape}")
   

