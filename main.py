import argparse
import os
from datetime import datetime


from hehrgnn_model import hehrgnn_model
from data_input import InputDataManager
from training import Training
from evaluation import EvaluatorClass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HEHR")

    parser.add_argument('-dataset', type=str, default="wd50k_unified_format")
    parser.add_argument('-max_arity', type=int, default=30) 
    parser.add_argument('-max_q_pairs', type=int, default=20) 
    #wd50k dataset was found to have max_qpairs of 20  

    parser.add_argument('-neg_ratio', type=int, default=10)
    parser.add_argument('-emb_dim', type=int, default=100)
    parser.add_argument('-inductive', type=int, default=0)
    parser.add_argument('-shallow_embed', type=int, default=0)
    parser.add_argument('-use_edge_embed', type=int, default=1)
    
    
    #GNN parameters
    parser.add_argument('-num_gnn_layers',type=int,default=2)
   
    #Decoders implemented are distmult, hype
    parser.add_argument('-decoder', type=str, default="distmult")
    parser.add_argument('-hype_in_channels', type=int,default=1)
    parser.add_argument('-hype_out_channels', type=int,default=6)
    parser.add_argument('-hype_stride', type=int,default=2)
    parser.add_argument('-hype_filt_w', type=int, default=1)
    parser.add_argument('-hype_filt_h', type=int, default=1)
    parser.add_argument('-hype_hidden_drop',type=float, default=0.2) 

    parser.add_argument('-optimizer', type=str, default="adam")
    parser.add_argument('-loss',type=str, default="BCEL") #BCEL is the default for LP
    #and CE is the default for NC. We can also use BCE for NC but then we need to 
    parser.add_argument('-epochs', type=int,default=20)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-learning_rate', type=float,default=0.001)    #tried 0.0001

    parser.add_argument('-drop_prob', type=float,default=0.3)
    parser.add_argument('-use_bn', type=int, default=0)

    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-output_dir', type=str, default="chkpnts")
    parser.add_argument('-run_mode', type=str, default="train") # train or eval
    parser.add_argument('-load_model', type=str, default="")
    parser.add_argument('-task', type=str, default="link_prediction")   #link_prediction or node_classification


    



    
    

    config = parser.parse_args()

    if (config.dataset == "cora"):
        config.max_arity = 2
        config.max_q_pairs = 0

    dataset = InputDataManager(config)
    print(f"Dataset loaded ...{dir(dataset)}...\ncontains {dataset.graph_representation.keys()} .....\nNum ents = {len(dataset.ent2id)} \n Num rels = {len(dataset.rel2id)} \n Num edges = {max(dataset.graph_representation['edge_index'][1])}")

    model = hehrgnn_model(config,dataset).to(config.device)

    if config.load_model !="":
        saved_model = os.path.join(config.output_dir, config.load_model)
        model.load_model(saved_model)

    #Now we have the embeddings for entities as well as relations. How do we evaluate?
    #The encoder decoder should be part of the GraphEncoder in order to optimize
    #  the embedding based on the reconstruction loss?
    #Add the loss functions
    #What is the default optimizer used by nn.Modules? If we don't specify any optimizer
    #for GraphEncoder, does it use a default one? Even for GNNlayer?
    #Where do we put the optimizer, it should be 
    #we cal it loss_layer because , the loss fn is implemented as a layer in torch.nn
    print(">>>>>>>>>Model params: ",sum([param.nelement() for param in model.parameters()]))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"name={name}, param.data = {param.shape}")

    if (config.run_mode=="train"):
        trainer = Training(model,config)        
        trainer.training_loop(dataset)
    
      
    
    # Final testing of the model - after training or loading a pre-trained model
    #log_file_name = os.path.join(config.output_dir, f"{config.dataset}_{config.decoder}_{config.loss}_embdim-{config.emb_dim}_epochs-{config.epochs}_numgnn-{config.num_gnn_layers}_BN-{config.use_bn}_induct-{config.inductive}_use_edgeEmb-{config.use_edge_embed}_shalEmb-{config.shallow_embed}_test_log.txt")  
    log_file_name = os.path.join("results",f"{(datetime.now()).strftime('%Y%m%d_%H%M%S')}_{dataset.name}_{config.decoder}_{config.loss}_embdim-{config.emb_dim}_epochs-{config.epochs}_numgnn-{config.num_gnn_layers}_BN-{config.use_bn}_induct-{config.inductive}_use_edgeEmb-{config.use_edge_embed}_shalEmb-{config.shallow_embed}_test_log.txt")

    print(">>>>>>>>>Loaded Model params: ",sum([param.nelement() for param in model.parameters()]))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"name={name}, param.data = {param.shape}")

   
    evaluator = EvaluatorClass(model,dataset,config)


    log_file = open(log_file_name, "a")
    metrics = evaluator.evaluate(split="test")
    log_file.write(f"Hyper parameters: {config} \n")
    log_file.write(f"Final test Evaluation Metrics {metrics} \n")
    log_file.close()

