import torch
from utils import get_param
from torch.nn.init import xavier_normal_,xavier_uniform_

from gnn_layer import GNNLayer
from graph_decoder import DistMultDecoder,HypEDecoder

#This is the encoder that generates embeddings for the nodes/edges in the input graph. 
# This in turn calls the gnn_layer. for message propagation among the nodes/edges for graph convolution.
# We can have multiple gnn_layers and configure suitable number of layers, embedding dimensions etc.

class GraphEncoder(torch.nn.Module):
    def __init__(self, dataset,config):
        super().__init__()

        self.config = config
        self.device = config.device
        self.num_gnn_layers = config.num_gnn_layers
        self.emb_dims = torch.full((self.num_gnn_layers + 1,), config.emb_dim) 

        
        self.num_ent = dataset.num_entities
        self.num_rel = dataset.num_relations

        self.task = config.task


        if self.task == "node_classification":
            self.out_dim = dataset.data["labels"].max().item() + 1
            self.emb_dims[-1] = self.out_dim
        
        #if the node and edge feature matrices are passed as part of the dataset, then we can use them as the initial embeddings. Otherwise, we need to generate initial embeddings.
       
        if (config.inductive == 0):
            if dataset.data["node_features"]  is not None:
                print(f">>>>>>Using the node features passed as part of the dataset as initial entity embeddings")
                self.emb_dims[0]= dataset.data["node_features"].shape[1]
                self.init_ent_embed = dataset.data["node_features"].to(self.device)
            else:
                self.init_ent_embed = get_param((self.num_ent, self.emb_dims[0]), 1) 

            self.init_rel_embed = get_param((self.num_rel, self.emb_dims[0]), 1)
        else:
            print(f">>>>>>Its running in Inductive mode>>>")
            #self.init_ent_embed = torch.ones(self.num_ent, self.emb_dim1, device=self.device)
            #self.init_rel_embed = torch.ones(self.num_rel, self.emb_dim1, device=self.device) 
            self.init_ent_embed = torch.full((self.num_ent, self.emb_dims[0]),0.3, device=self.device)
            self.init_rel_embed = torch.full((self.num_rel, self.emb_dims[0]),0.3, device=self.device) 
            print(f"$$$$$ Graph encoder init embedd={self.init_ent_embed} rel embed={self.init_rel_embed}")

        """ self.register_buffer('init_ent_embed', torch.randn(self.num_ent, self.emb_dim1))
        #self.ent_embed = torch.randn(self.num_ent, self.emb_dim1)
        xavier_normal_(self.ent_embed)


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
        
        # Shallow embedding model just optimizes lookup tables for the entities and relations without any message propagation i.e it does not learn any layer weights. Hence the forward pass is just a lookup and score calculation with the decoder.
        if config.shallow_embed == 0:
            # Why do we need a separate class for encoder? Can it be not part of the GNNLayer itself?
            self.gnn_layer1 = GNNLayer(self.emb_dims[0], self.emb_dims[1],self.device,self.config)
            if self.gnn_layer1: self.gnn_layer1.to(self.device)

            #Instead of having multiple layers defined here, can we have a single layer
            #with multiple propagations? But then we'll have only a single set of weights? Is that ok?
            self.gnn_layer2 = GNNLayer(self.emb_dims[1], self.emb_dims[2],self.device,self.config)
            if self.gnn_layer2: self.gnn_layer2.to(self.device)

            if(self.num_gnn_layers > 2):
                self.gnn_layer3 = GNNLayer(self.emb_dims[2], self.emb_dims[3],self.device,self.config)
                if self.gnn_layer3: self.gnn_layer3.to(self.device)
            if(self.num_gnn_layers > 3):
                self.gnn_layer4 = GNNLayer(self.emb_dims[3], self.emb_dims[4],self.device,self.config)
                if self.gnn_layer4: self.gnn_layer4.to(self.device)


            self.hidden_drop1 = torch.nn.Dropout(config.drop_prob)
            self.hidden_drop2 = torch.nn.Dropout(config.drop_prob)
            self.hidden_drop3 = torch.nn.Dropout(config.drop_prob)
            self.hidden_drop4 = torch.nn.Dropout(config.drop_prob)
        
       


        

        #Decoders
        match config.decoder:
            case "distmult":
                self.decoder = DistMultDecoder(dataset,config)
            case "hype": 
                self.decoder = HypEDecoder(dataset,config)
        
        #self.bias = get_param((self.num_ent), 0)

    def load_model(self,saved_model):
        self.load_state_dict(torch.load(saved_model))
        print(f" Saved Models is loaded....")
        if torch.cuda.is_available():
            allocated_memory_bytes = torch.cuda.memory_allocated()
            print(f"Memory allocated by tensors: {allocated_memory_bytes / (1024**2):.2f} MB")
            free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info()
            print(f"Free GPU memory: {free_memory_bytes / (1024**2):.2f} MB")
            print(f"Total GPU memory: {total_memory_bytes / (1024**2):.2f} MB")
   
    def forward(self,graph_data,task,input_for_pred):
        #input_for_pred contains (rel_idx,ent1_idx,ent2_idx,ent3_idx,ent4_idx,ent5_idx,ent6_idx,pres_bits,abs_bits)      
        #Should we make ent_embed and rel_embed as registered buffers so that they are 
        #also saved as part of the model, but are not optimized with grad descent
       #The init_ent_embed is getting optimized over the iterations

       #!!!Trying with an forward time init embed buffer that is part of the model params hence not optimized

       
        if self.config.shallow_embed == 0:
            #print(f"Xavier inited embed={self.init_ent_embed} rel embed={self.init_rel_embed}")
            ent_embed, rel_embed = self.gnn_layer1(self.init_ent_embed,self.init_rel_embed,graph_data)
            #drop reqd?
            ent_embed = self.hidden_drop1(ent_embed)
            #self.rel_embed = drop1(self.rel_embed)
            ent_embed, rel_embed = self.gnn_layer2(ent_embed,rel_embed,graph_data)
            #Should the Dropout be moved inside the if condition for num_gnn_layers? 
            # f num_gnn_layers is 1, then we should not have the hidden_drop2 layer in use. Hence it should be inside the if condition for num_gnn_layers>1
            if(self.num_gnn_layers > 2):
                ent_embed = self.hidden_drop2(ent_embed)
                ent_embed, rel_embed = self.gnn_layer3(ent_embed,rel_embed,graph_data)

            if(self.num_gnn_layers > 3):
                ent_embed = self.hidden_drop3(ent_embed)
                ent_embed, rel_embed = self.gnn_layer4(ent_embed,rel_embed,graph_data)
                #ent_embed = self.hidden_drop4(ent_embed)
            #drop reqd?
            #self.rel_embed = drop2(self.rel_embed)            
        else:
            ent_embed = self.init_ent_embed
            rel_embed = self.init_rel_embed

        if task == "link_prediction":
            score = self.decoder(ent_embed, rel_embed,input_for_pred)
            #score = self.dm_decoder(ent_embed, rel_embed,rel_idx,ent1_idx,ent2_idx,ent3_idx,ent4_idx,ent5_idx,ent6_idx,pres_bits,abs_bits)
            #score = self.hype_decoder(ent_embed,rel_embed,rel_idx,ent1_idx,ent2_idx,ent3_idx,ent4_idx,ent5_idx,pres_bits,abs_bits)
        else:
            score = ent_embed
           
        return score

    




