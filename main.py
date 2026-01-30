import argparse
import os


from graph_encoder import GraphEncoder
from data_input import InputDataManager
from training import Training
from evaluation import EvaluatorClass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HEHR")

    parser.add_argument('-dataset', type=str, default="wd50k_unified_format")
    parser.add_argument('-max_arity', type=str, default=6) 
    parser.add_argument('-max_q_pairs', type=str, default=20) 
    #wd50k dataset was found to have max_qpairs of 20  

    parser.add_argument('-neg_ratio', type=int, default=10)
    parser.add_argument('-emb_dim', type=int, default=100)
    parser.add_argument('-inductive', type=int, default=0)
    parser.add_argument('-shallow_embed', type=int, default=0)
    parser.add_argument('-edge_embed', type=int, default=1)
    
    
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
    parser.add_argument('-loss',type=str, default="BCEL")
    parser.add_argument('-epochs', type=int,default=20)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-learning_rate', type=float,default=0.001)    #tried 0.0001

    parser.add_argument('-drop_prob', type=float,default=0.3)
    parser.add_argument('-use_bn', type=int, default=0)

    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-output_dir', type=str, default="chkpnts")
    parser.add_argument('-run_mode', type=str, default="train") # train or eval
    parser.add_argument('-load_model', type=bool, default=False)


    



    
    

    config = parser.parse_args()
    dataset = InputDataManager(config)
    print(f"Dataset loaded ...{dir(dataset)}...\ncontains {dataset.graph_representation.keys()} .....\nNum ents = {len(dataset.ent2id)} \n Num rels = {len(dataset.rel2id)} \n Num edges = {len(dataset.graph_representation['edge_index'])}")

    graph_encoder = GraphEncoder(dataset,config).to(config.device)

    if config.load_model==True:
        saved_model = os.path.join(config.output_dir, '20251209_201441_wd50k_unified_format_distmult_BCEL_embdim-128_epochs-50_transductive_best_model.chkpnt')
        graph_encoder.load_model(saved_model)

    #Now we have the embeddings for entities as well as relations. How do we evaluate?
    #The encoder decoder should be part of the GraphEncoder in order to optimize
    #  the embedding based on the reconstruction loss?
    #Add the loss functions
    #What is the default optimizer used by nn.Modules? If we don't specify any optimizer
    #for GraphEncoder, does it use a default one? Even for GNNlayer?
    #Where do we put the optimizer, it should be 
    #we cal it loss_layer because , the loss fn is implemented as a layer in torch.nn
    print(">>>>>>>>>Graph Encoder Model params: ",sum([param.nelement() for param in graph_encoder.parameters()]))
    for name, param in graph_encoder.named_parameters():
        if param.requires_grad:
            print(f"name={name}, param.data = {param.shape}")
   
    if (config.run_mode=="train"):
        trainer = Training(graph_encoder,config)        
        trainer.training_loop(dataset)
    else:
      
        evaluator = EvaluatorClass(graph_encoder,dataset,config)
        
    
        print(">>>>>>>>>Loaded Model params: ",sum([param.nelement() for param in graph_encoder.parameters()]))
        for name, param in graph_encoder.named_parameters():
            if param.requires_grad:
                print(f"name={name}, param.data = {param.shape}")

        metrics = evaluator.evaluate()     
   
   

