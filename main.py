import argparse


from graph_encoder import GraphEncoder
from data_input import InputDataManager
from training import Training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HEHR")

    parser.add_argument('-dataset', type=str, default="wd50k_unified_format")
    parser.add_argument('-max_arity', type=str, default=6) 
    parser.add_argument('-max_q_pairs', type=str, default=20) 
    #wd50k dataset was found to have max_qpairs of 20  

    parser.add_argument('-neg_ratio', type=int, default=5)
    parser.add_argument('-emb_dim', type=int, default=100)
   
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
    parser.add_argument('-epochs', type=int,default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-learning_rate', type=float,default=0.001)
    parser.add_argument('-drop_prob', type=float,default=0.3)

    parser.add_argument('-device', type=str, default="cuda")



    
    #tried 0.0001
    

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
    #we cal it loss_layer because , the loss fn is implemented as a layer in torch.nn
    trainer = Training(graph_encoder,config)
    trainer.training_loop(dataset)
    
    
    print(">>>>>>>>>Graph Encoder Model params: ",sum([param.nelement() for param in graph_encoder.parameters()]))
    for name, param in graph_encoder.named_parameters():
        if param.requires_grad:
            print(f"name={name}, param.data = {param.shape}")
   

