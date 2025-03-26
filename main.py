import argparse
import torch


from graph_encoder import GraphEncoder
from data_input import InputDataManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HEHR")
    parser.add_argument('-dataset', type=str, default="wd50k_new_format")
    parser.add_argument('-max_arity', type=str, default=6) 
    parser.add_argument('-max_q_pairs', type=str, default=20)   
     
    parser.add_argument('-emb_dim', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-device', type=str, default="cuda")

    args = parser.parse_args()
    dataset = InputDataManager(args)
    print(f"Dataset loaded ...{dir(dataset)}...\ncontains {dataset.graph_representation.keys()} .....\nGraph Repsn{len(dataset.ent2id)}")

    graph_encoder = GraphEncoder(dataset,args).to(args.device)
    graph_encoder(dataset.graph_representation["edge_index"], dataset.graph_representation["edge_type"])

    #Now we have the embeddings for entities as well as relations. How do we evaluate?
    #The encoder decoder should be part of the GraphEncoder in order to optimize
    #  the embedding based on the reconstruction loss?
    #Add the loss functions
    #What is the default optimizer used by nn.Modules? If we don't specify any optimizer
    #for GraphEncoder, does it use a default one? Even for GNNlayer?
    #Where do we put the optimizer, it should be 