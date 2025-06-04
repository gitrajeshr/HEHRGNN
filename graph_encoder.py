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
        self.emb_dim2 = 100
        self.emb_dim3 = 100
        self.init_ent_embed = get_param((self.num_ent, self.emb_dim1), 1) 
        self.init_rel_embed = get_param((self.num_rel, self.emb_dim1), 1) 
        self.bias = get_param((self.num_ent), 0) 


        """ self.register_buffer('init_ent_embed', torch.randn(self.num_ent, self.emb_dim1))
        #self.ent_embed = torch.randn(self.num_ent, self.emb_dim1)
        xavier_normal_(self.ent_embed)

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

        self.register_buffer('_init_rel_embed', torch.randn(self.num_rel, self.emb_dim1))
        #self.rel_embed = torch.randn(self.num_rel, self.emb_dim1) # * 2 is for inverse relns
        xavier_normal_(self.rel_embed) """


        """ phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim1 // 2)
        self.rel_embed = nn.Parameter(torch.cat([
                torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            ], dim=0)) """
       

        #self.rel_embed = torch.randn(self.num_rel, self.emb_dim1) # * 2 is for inverse relns
        
        #self.init_embed.data[0] = 0  # padding  ---- Why is this required??
        
        # Why do we need a separate class for encoder? Can it be not part of the GNNLayer itself?
        self.gnn_layer1 = GNNLayer(self.emb_dim1, self.emb_dim2,self.device)
        #Instead of having multiple layers defined here, can we have a single layer
        #with multiple propagations? But then we'll have only a single set of weights? Is that ok?
        self.gnn_layer2 = GNNLayer(self.emb_dim2, self.emb_dim1,self.device)

        if self.gnn_layer1: self.gnn_layer1.to(self.device)
        if self.gnn_layer1: self.gnn_layer1.to(self.device)


    
    def predict_entity(self, ent_embed, rel_embed,rel,sub):
        sub_emb = torch.index_select(ent_embed, 0, sub)
        rel_emb = torch.index_select(rel_embed, 0, rel)
        #Pred is how the masked entity embedding os predicted from rel and ent1 embedding

        #ConvE decoder
        """  stk_inp			= self.concat(sub_emb, rel_emb)
        pred				= self.bn0(stk_inp)
        pred				= self.m_conv1(pred)
        pred				= self.bn1(pred)
        pred				= F.relu(pred)
        pred				= self.feature_drop(pred)
        pred				= pred.view(-1, self.flat_sz)
        pred				= self.fc(pred)
        pred				= self.hidden_drop2(pred)
        pred				= self.bn2(pred)
        pred				= F.relu(pred)

        pred = torch.mm(pred, ent_embed.transpose(1,0))
        pred += self.bias.expand_as(pred) """

        #DIstMult decoder
        pred = sub_emb * rel_emb
        pred = torch.mm(pred, ent_embed.transpose(1, 0))
        pred += self.bias.expand_as(pred)

        #In StarE Transformer is used as the decoder for predicting the entity embedding
        #The score generated is as a probability over all the entities
        print(f"^^^^^^^^^^In Predict entity Max={torch.max(pred)} pred= {pred}")
        #score = torch.sigmoid(pred)
        score = pred
        return score
    def forward(self,graph_data,rel,ent1):
        #Should we make ent_embed abd rel_embed as registered buffers so that they are 
        #also saved as part of the model, but are not optimized with grad descent
        ent_embed, rel_embed = self.gnn_layer1(self.init_ent_embed,self.init_rel_embed,graph_data)
        #drop reqd?
        #self.ent_embed = drop1(self.ent_embed)
        #self.rel_embed = drop1(self.rel_embed)
        self.ent_embed, self.rel_embed = self.gnn_layer2(ent_embed,rel_embed,graph_data)
        #drop reqd?
        #self.ent_embed = drop2(self.ent_embed)
        #self.rel_embed = drop2(self.rel_embed)
        score = self.predict_entity(ent_embed, rel_embed,rel,ent1)
        return score

    




