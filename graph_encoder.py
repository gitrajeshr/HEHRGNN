import torch
from utils import get_param
from torch.nn.init import xavier_normal_


from gnn_layer import GNNLayer

#This is the encoder that generates embeddings for the nodes/edges in the input graph. 
# This in turn calls the gnn_layer. for message propagation among the nodes/edges for graph convolution.
# We can have multiple gnn_layers and configure suitable number of layers, embedding dimensions etc.

class GraphEncoder(torch.nn.Module):
    def __init__(self, dataset,config):
        super().__init__()

        self.device = config.device

        self.num_ent = dataset.num_entities
        self.num_rel = dataset.num_relations
        self.emb_dim1 = 100
        self.emb_dim2 = 128
        self.emb_dim3 = 200
        self.register_buffer('ent_embed', torch.randn(self.num_ent, self.emb_dim1))
        #self.ent_embed = torch.randn(self.num_ent, self.emb_dim1)
        xavier_normal_(self.ent_embed)

        """ phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim1 // 2)
        self.rel_embed = nn.Parameter(torch.cat([
                torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            ], dim=0)) """
        self.register_buffer('rel_embed', torch.randn(self.num_ent, self.emb_dim1))
        #self.rel_embed = torch.randn(self.num_rel, self.emb_dim1) # * 2 is for inverse relns
        xavier_normal_(self.rel_embed)

        #self.rel_embed = torch.randn(self.num_rel, self.emb_dim1) # * 2 is for inverse relns
        
        #self.init_embed.data[0] = 0  # padding  ---- Why is this required??
        
        # Why do we need a separate class for encoder? Can it be not part of the GNNLayer itself?
        self.gnn_layer1 = GNNLayer(self.emb_dim1, self.emb_dim2)
        #Instead of having multiple layers defined here, can we have a single layer
        #with multiple propagations? But then we'll have only a single set of weights? Is that ok?
        self.gnn_layer2 = GNNLayer(self.emb_dim2, self.emb_dim3)

        if self.gnn_layer1: self.gnn_layer1.to(self.device)
        if self.gnn_layer1: self.gnn_layer1.to(self.device)
    def forward(self,edge_index, edge_type):
        self.ent_embed, self.rel_embed = self.gnn_layer1(self.ent_embed,self.rel_embed,edge_index, edge_type)
        #drop reqd?
        self.ent_embed, self.rel_embed = self.gnn_layer2(self.ent_embed,self.rel_embed,edge_index, edge_type)
        #drop reqd?
        return self.ent_embed, self.rel_embed

    




