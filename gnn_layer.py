import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree
from utils import get_param
from torch_geometric.utils import scatter, softmax
from torch import Tensor





class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,device):
        super().__init__(aggr='add', flow='source_to_target',node_dim=0)  #node_dim=0??
        self.lin = Linear(in_channels, in_channels, bias=False)
        #self.bias = Parameter(torch.empty(out_channels))
        self.set_weights(in_channels, out_channels)
        self.device= device
        self.config = {}
        self.config['QUAL_TRANSFORM']  = 'mult' #'sub', 'corr','rotate', '
        self.config['QUALS_AGGREGATE']  = 'sum' #'mean'
        self.config['EDGE_QUALS_AGGREGATE']  = 'sum' #'conact', 'mult'
        self.config['QUALS_WEIGHT'] = 0.2

        


    def set_weights(self,in_channels, out_channels):
        self.lin.reset_parameters()
        #self.bias.data.zero_()
        self.w_nodes = get_param((in_channels, in_channels))  
        self.w_edges = get_param((in_channels, out_channels))  
        self.w_rel = get_param((in_channels, out_channels))  
        self.w_quals = get_param((in_channels, out_channels))
        print(f"In weight w_nodes shape={self.w_nodes.shape} w_edges shape={self.w_edges.shape}")

    
    def forward(self, ent_embed,rel_embed, graph_data,hyperedge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        print(f"dataset={graph_data} keys={graph_data.keys()}")
        hyperedge_index = graph_data["edge_index"]
        edge_type = graph_data["edge_type"]
        qual_details = graph_data["qual_details"]

        num_nodes = ent_embed.size(0)

        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        #x.new_ones(size) creates a tensor of size, "size" and of dtype and device
        #same as that of x
        if hyperedge_weight is None:
            hyperedge_weight = ent_embed.new_ones(num_edges)

        
        #Below scatter stmt computes degrees of nodes by scattering the edge weight into
        # an output tensor by using the node members of hyperedges as indices for the output
        #.eg, if nodes 1,2,3 are members of hyperedge 1, the edge weight of edge 1 is transmitted
        # to the three indices 1,2,3 in the output. So effectively each index in the output
        # gets contributions from all the edges of which it is a part
        #  D -> the node degrees

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0
        #Below scatter stmt computes the degree of edges by scattering 1's to the indices
        # pointed by the second row of hyper_edge_index i.e. edge index repeated as many times as the number of nodes
        #  B -> the edge degrees
        B = scatter(ent_embed.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=ent_embed.size(0))

        # Step 2: Linearly transform node feature matrix.
        print(f"Before lin transform  ent_embed={ent_embed.shape} B={B}")
        ent_embed = self.lin(ent_embed) #??? why its is reqd??
        print(f"  ")
        print(f"After lin transform num_nodes ={num_nodes} num_edges={num_edges} hyperedge_index={hyperedge_index.shape} Edge type shape = {edge_type.shape} values={edge_type} \n ent_embed={ent_embed.shape} rel_embed={rel_embed.shape}\n rel_embed={rel_embed}")
        edge_embed = torch.zeros_like(rel_embed[edge_type])
        ent_embed = torch.ones_like(ent_embed)
        #propagate message using the MessagePassing class features
        #for hyperedges, the propagation using MessagePassing class has to be done in two steps
        #first propagate messages from the member nodes to the hyperedges and update the edge emebddings
        #Then propagate from the hyperedges to the member nodes and update the node embeddings
       
       ##!!The arguments to propagate should be such that, if size=(N,M) and , if x_i is accessed inside message
       #then x has to be of dimension (M,..) and if x_j is accessed then x has to be of dim (N,..)
        msg_dirn = 0
        out_edge_embed = self.propagate(hyperedge_index,size=(num_nodes,num_edges),x=(ent_embed,edge_embed),msg_dirn=msg_dirn, norm=B, qual_details =None, weight=self.w_nodes)

        print(f"AFter 1st propagation shape={out_edge_embed.shape} hyperedge_index.flip([0]) shape")
        msg_dirn = 1
        #qual_details = torch.full((num_nodes, 3), 3, device=self.device)
        #???Prepare qual_details as indexed by egdesi.e. of dimension(num_edges, ..)
        out_ent_embed = self.propagate(hyperedge_index.flip([0]),size=(num_edges,num_nodes), x=(out_edge_embed,ent_embed), msg_dirn=msg_dirn, qual_details = qual_details,norm=D,weight = self.w_edges)
        
        print(f"AFter 2nd propagation shape={out_ent_embed.shape}")

        #In case of hyper  edges, the relation embeddings have to be updated by propagation from the 
        # neighboring(or member nodes). Need to do it here or Some where else???
        out_rel_embed = torch.matmul(rel_embed, self.w_rel) # what transformation is to be applied??
        print(f"Returning from GNN layer  ent_embed shape={out_ent_embed.shape}")

        return out_ent_embed, out_rel_embed
    
    def update_hyperedge_with_nodes(self,edge_embed,obj_embed, weight):
            print(f"obj_embed={obj_embed.shape} rel_embed={edge_embed.shape}")
            # if transform is mult
            trans_embed = obj_embed * edge_embed
            #else
            edge_msg = torch.einsum('ij,jk->ik', trans_embed, weight) #W_lambda(r)
            return edge_msg
    
    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """

        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            aggregated_qual_embed          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

        
        """

        if self.config['QUALS_AGGREGATE'] == 'sum':
            aggregated_edge_qual_embed = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
        elif self.config['QUALS_AGGREGATE'] == 'mean':
            aggregated_edge_qual_embed = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = aggregated_edge_qual_embed.sum(dim=-1) == 0
            aggregated_edge_qual_embed[mask] = fill

        return aggregated_edge_qual_embed

    def qualifier_aggregate(self, edge_embed, qualifier_embed, quals_index):
        """
           Step1 : 
           Aggregate the qualifier embeddings for all the qualifiers of a edge using
            self.coalesce_quals
           step 2:
           multiply the aggregated qualifiers embedding with a weight and combine 
           it with the edge embedding based on defined aggregation strategy ("'sum', 'concat', 'mult' etc.)
        """
        #!!!Need to apply the weights properlys
        if self.config['EDGE_QUALS_AGGREGATE'] == 'sum':
            edge_qual_embed = torch.einsum('ij,jk -> ik',
                                         self.coalesce_quals(qualifier_embed, quals_index, edge_embed.shape[0]),
                                         self.w_quals)
            quals_weight = self.config['QUALS_WEIGHT']
            edge_embed_with_quals =  (1-quals_weight) * edge_embed + quals_weight * edge_qual_embed      # [N_EDGES / 2 x EMB_DIM]
        elif self.config['EDGE_QUALS_AGGREGATE'] == 'concat':
            edge_qual_embed = self.coalesce_quals(qualifier_embed, quals_index, edge_embed.shape[0])
            edge_embed_with_quals = torch.cat((edge_embed, edge_qual_embed), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            edge_embed_with_quals =  torch.mm(edge_embed_with_quals, self.w_quals)                         # [N_EDGES / 2 x EMB_DIM]

        elif self.config['EDGE_QUALS_AGGREGATE'] == 'mul':
            edge_qual_embed = torch.mm(self.coalesce_quals(qualifier_embed, quals_index, edge_embed.shape[0], fill=1), self.w_q)
            edge_embed_with_quals =  edge_embed * edge_qual_embed
        else:
            raise NotImplementedError    
        
        return edge_embed_with_quals
    
    def qual_transform(self, qualifier_ent, qualifier_rel):
        if self.config['QUAL_TRANSFORM'] == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.config['QUAL_TRANSFORM'] == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.config['QUAL_TRANSFORM'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.config['QUAL_TRANSFORM'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return trans_embed 
    def update_hyperedge_with_quals(self,rel_embed,ent_embed,edge_type,edge_qual_details):
            print(f"edge_qual_details={edge_qual_details} rel_embed={edge_embed.shape}")
            
            qualifier_rel = edge_qual_details[0]
            qualifier_ent = edge_qual_details[1]
            qualifier_index = edge_qual_details[2]
            
            # Step 1: Retrieve embeddings for qual ent/relns
            qual_rel_embed = rel_embed[qualifier_rel]
            qual_ent_embed = ent_embed[qualifier_ent]

            #Step 2 Retrieve reln embeddings for the edges
            edge_embed = rel_embed[edge_type]
            print(f"In update_hyperedge_with_quals = {rel_embed.shape,edge_type.shape,edge_embed.shape}")

            # Step 3: generate the qualifier embedding by combing ent/rel embeddings
            qualifier_embed = self.qual_transform(qualifier_ent=qual_ent_embed,
                                                qualifier_rel=qual_rel_embed)

            # Aggregate the qualifiers for each edge

            edge_embed_with_quals = self.qualifier_aggregate(edge_embed, qualifier_embed, qualifier_index)

            return edge_embed_with_quals
    
    def message(self, edge_index,size,x,x_j,x_i,msg_dirn,norm_i, qual_details, weight=None):
        #norm_i=[]
        print(f"Edge_index= norm_i={norm_i} x shape={x[0].shape, x[1].shape} x_j={x_j.shape} x_i={x_i.shape} x_i={x_i} x_j ={x_j}")
        #Hyper edges with qualifiers - Hyperedge Hyperrelational graphs
        #1)all qualifiers of an edge are combined
        # 2) update hyperedge embeddings with qualifiers embedding and the member node embeddings
        # 3)The node embeddings are updated using the neighbouring nodes and hyper edge emebddings

        #weight = getattr(self, 'w_{}'.format(mode))
        #Update the rel_embed using the qualifier pairs and then prepare the edge msg
        #using the x_j and the rel_embed
        #?? Or should we do combining quals with edges before calling propagate? 
        if msg_dirn==0: # from nodes to edge    
            #do we need to combine quals with edges before propagation from nodes?
            edge_embed_with_quals = self.update_hyperedge_with_quals(x_j, ent_embed, rel_embed, qual_details) #(edge_embed, quals)
            msg = self.update_hyperedge_with_nodes(x_j,edge_embed_with_quals,weight) #(node_embed, edge_embed)
        else: #from edges to nodes
            msg = self.update_nodes_with_edges(x_j,weight) #(edge_embed, quals)
            #msg = self.edge_msg_transform(x_j, edge_embed) #Phi_r in StaRE equation

        #x_j , is the source node which sends the message. WHich here is the source node or 1st node in the edge representation
        #We need to send it to the hyperedge , where it will be aggregated. each hyperedge gets messages
        # from all the nodes which are members of it to the member nodes.
        #x_i is the target node which receives the message(here the hyperedges )
        #in the second call, propagation happens in the other direction, where we send 
        # the hyperedge feature to the member nodes. Each member node gets message from 
        # all the hyperedges of which it is a member

         #So where do we use the relation embedding?
         #when a hyperedge receives messsages from its member nodes and aggregates, may be 
         #we can include the relation embedding in the aggregation with some weightage/attention
        print(f"in message x_j shape={x_j.shape}")
        #out = norm_i.view(-1, 1,1) * x_j.view(-1, H, F)

        out = torch.einsum('ij,jk->ik', msg, weight)
        #out = x_j
        #out is the matrix containing all the message rows corresponding to x_j for all the edges
        #in the edge_index. Now these out messages will be aggregated for all the x_i (here the hyperedges)
        #using the aggregate function(default is "add") and the x_i is updated using the update function(default is none)

        #Norm is the edge degree in first call and node degree in second call of propagate.
        # Hence multiplying out with norm_i normailzes out by the degree of of edge/node respectively
        #in the first and second call
        return out if norm_i is None else out * norm_i.view(-1, 1)


        return out
