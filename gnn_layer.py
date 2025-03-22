import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree
from utils import get_param



class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow='source_to_target')  #node_dim=0??
        self.lin = Linear(in_channels, in_channels, bias=False)
        #self.bias = Parameter(torch.empty(out_channels))
        self.set_weights(in_channels, out_channels)
        print(f"In channel ={in_channels} out_channels={out_channels}")
        self.device = "cpu"

    def set_weights(self,in_channels, out_channels):
        self.lin.reset_parameters()
        #self.bias.data.zero_()
        self.W_msg = get_param((in_channels, out_channels))  
    
    def forward(self, ent_embed,rel_embed, edge_index,edge_type):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=ent_embed.size(0))

        # Step 2: Linearly transform node feature matrix.
        print(f"Before lin transform  ent_embed={ent_embed.shape}")
        ent_embed = self.lin(ent_embed) #??? why its is reqd??
        print(f"  ")
        print(f"After lin transform Edge type shape = {edge_type.shape} values={edge_type} \n ent_embed={ent_embed.shape} rel_embed={rel_embed.shape}\n rel_embed={rel_embed}")
        edge_embed = rel_embed[edge_type]
        #propagate message using the MessagePassing class features
        out_ent_embed = self.propagate(edge_index, x=ent_embed,edge_embed=edge_embed)
        out_rel_embed = rel_embed # what transformation is to be applied??
        return out_ent_embed, out_rel_embed
    
    def edge_msg_transform(self,obj_embed, edge_embed, weight):
            print(f"obj_embed={obj_embed.shape} rel_embed={edge_embed.shape}")
            # if transform is mult
            trans_embed = obj_embed * edge_embed
            #else
            edge_msg = torch.einsum('ij,jk->ik', trans_embed, weight) #W_lambda(r)
            return edge_msg
    
    def message(self, edge_index,x,x_j, x_i, edge_embed):
        print(f"Edge_index={edge_index.shape} x={x.shape} x_j={x_j.shape} x_i={x_i.shape}")
        #Hyper edges with qualifiers - Hyperedge Hyperrelational graphs
        #1)all qualifiers of an edge are combined
        # 2) update hyperedge embeddings with qualifiers embedding and the member node embeddings
        # 3)The node embeddings are updated using the neighbouring nodes and hyper edge emebddings

        #weight = getattr(self, 'w_{}'.format(mode))
        #Update the rel_embed using the qualifier pairs and then prepare the edge msg
        #using the x_j and the rel_embed
        weight = self.W_msg
        xj_msg = self.edge_msg_transform(x_j, edge_embed, weight) #Phi_r in StaRE equation
        return xj_msg
