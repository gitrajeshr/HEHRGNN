import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow='source_to_target')  #node_dim=0??
        self.lin = Linear(in_channels, out_channels, bias=False)
        #self.bias = Parameter(torch.empty(out_channels))
        self.set_weights()

    def set_weights(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, ent_embed,rel_embed, edge_index,edge_type):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x) #??? why its is reqd??

        #propagate message using the MessagePassing class features
        out_ent_embed = self.propagate(edge_index, x=ent_embed, edge_type=edge_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                    qualifier_rel=self.in_index_qual_rel,
                                    qual_index=self.quals_index_in,
                                    source_index=self.in_index[0])
        out_rel_embed = rel_embed # what transformation is to be applied??
        return out_ent_embed, out_rel_embed
    
    def message(self, edge_index,x,x_j, x_i, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None, source_index=None):
        #Hyper edges with qualifiers - Hyperedge Hyperrelational graphs
        #1)all qualifiers of an edge are combined
        # 2) update hyperedge embeddings with qualifiers embedding and the member node embeddings
        # 3)The node embeddings are updated using the neighbouring nodes and hyper edge emebddings

    