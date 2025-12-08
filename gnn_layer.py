import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from torch_geometric.utils import add_self_loops, degree
from utils import get_param
from torch_geometric.utils import scatter, softmax
from torch import Tensor
import torch.nn.functional as F




class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,device):
        super().__init__(aggr='add', flow='source_to_target',node_dim=0)  #node_dim=0??
        self.ent_lin = Linear(in_channels, out_channels, bias=False)
        self.rel_lin = Linear(in_channels, out_channels, bias=False)
        #self.ent_bn = torch.nn.BatchNorm1d(out_channels)
        #self.rel_bn = torch.nn.BatchNorm1d(out_channels)
        #self.bias = Parameter(torch.empty(out_channels))
        self.set_weights(out_channels, out_channels)
        self.device= device
        self.config = {}
        self.config['QUAL_TRANSFORM']  = 'mult' #'sub', 'corr','rotate', '
        self.config['QUALS_AGGREGATE']  = 'sum' #'mean'
        self.config['EDGE_QUALS_AGGREGATE']  = 'sum' #'conact', 'mult'
        self.config['QUALS_WEIGHT'] = 0.2

        


    def set_weights(self,in_channels, out_channels):
        self.ent_lin.reset_parameters()
        self.rel_lin.reset_parameters()
        #self.bias.data.zero_()
        self.w_quals_to_edges= get_param((in_channels, out_channels), 1)
        self.w_nodes_to_edges = get_param((in_channels, in_channels),1)  
        self.w_edges_to_nodes = get_param((in_channels, out_channels),1)  
        self.w_edges_to_quals = get_param((in_channels, out_channels),1)  
        self.w_combine_edge_embed = Parameter(torch.tensor([0.33,0.33,0.33]))
        self.w_combine_ent_embed = Parameter(torch.tensor([0.33,0.33,0.33]))
        print(f"In weight w_nodes shape={self.w_nodes_to_edges.shape} w_edges shape={self.w_edges_to_nodes.shape}")
        #!!!For Testing Only. Remove it!!!!
        #self.w_rel = get_param((in_channels, out_channels), 1)  

    
    def forward(self, ent_embed,rel_embed, graph_data,hyperedge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        #print(f"dataset={graph_data} keys={graph_data.keys()}")
        hyperedge_index = graph_data["edge_index"]
        edge_type = graph_data["edge_type"]
        qual_details = graph_data["qual_details"]

        num_nodes = ent_embed.size(0)
        num_rels = rel_embed.size(0)
        # Step 2: Linearly transform node feature matrix.
        print(f"Before lin transform  ent_embed={ent_embed.shape}")
        
        ent_embed = self.ent_lin(ent_embed) #??? why its is reqd??
        rel_embed = self.rel_lin(rel_embed)
        print(f"  ")
        print(f"After lin transform ent_embed={ent_embed.shape} rel_embed={rel_embed.shape}")
        edge_embed = rel_embed[edge_type]
        #edge_embed = torch.zeros_like(rel_embed[edge_type])
        #ent_embed = torch.ones_like(ent_embed)
        


        num_edges = 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        #x.new_ones(size) creates a tensor of size, "size" and of dtype and device
        #same as that of x
        if hyperedge_weight is None:
            hyperedge_weight = ent_embed.new_ones(num_edges)
        #the edge_index details for the qual edges i.e src is qual node and dst is the hyperedge
        qual_edge_index = qual_details[[1,2]] 

        #Below scatter stmt computes degrees of nodes by scattering the edge weight into
        # an output tensor by using the node members of hyperedges as indices for the output
        #.eg, if nodes 1,2,3 are members of hyperedge 1, the edge weight of edge 1 is transmitted
        # to the three indices 1,2,3 in the output. So effectively each index in the output
        # gets contributions from all the edges of which it is a part
        
        #A -> qual_degree of an edge
        A = scatter(ent_embed.new_ones(qual_edge_index.size(1)), qual_edge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        A = 1.0/A
        A[A == float("inf")] = 0
        norm_qual_to_edge = A

        #  B -> the edge degrees
        B = scatter(ent_embed.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0
        norm_node_to_edge = B #edge_degree
       
        #  C -> primary node degrees

        C = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        C = 1.0 / C
        C[C == float("inf")] = 0
        norm_edge_to_node = C  #node _degree
        #Below scatter stmt computes the degree of edges by scattering 1's to the indices
        # pointed by the second row of hyper_edge_index i.e. edge index repeated as many times as the number of nodes
       
        #print(f"Qual edge index ={qual_edge_index.shape} {qual_edge_index}")
       
        # D -> qual nodes degree
        D = scatter(ent_embed.new_ones(qual_edge_index.size(1)), qual_edge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0/D
        D[D == float("inf")] = 0
        norm_edge_to_qual = D
        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=ent_embed.size(0))

        
        
        #propagate message using the MessagePassing class features
        #for hyperedges, the propagation using MessagePassing class has to be done in two steps
        #first propagate messages from the member nodes to the hyperedges and update the edge emebddings
        #Then propagate from the hyperedges to the member nodes and update the node embeddings
       
       ##!!Inside the message () function, x_j is the src node, x_i is the dst node
       # The arguments to propagate should be such that, if size=(N,M) and , if x_i is accessed inside message
       #then x has to be of dimension (M,..) and if x_j is accessed then x has to be of dim (N,..)

        #1 - quals to hyper edge

        #print(f"QQQQQQ>>>>qual_details = {qual_details.shape} qual edge index = {qual_edge_index.shape}")
        qual_edge_embed = rel_embed[qual_details[0]]
        #print(f"010101 Ent embed = {ent_embed}")
        msg_type = 1
        propagated_edge_msg1 = self.propagate(qual_edge_index,size=(num_nodes,num_edges),x=(ent_embed,edge_embed),msg_type=msg_type, norm=norm_qual_to_edge,qual_edge_embed=qual_edge_embed,  weight=self.w_quals_to_edges)
        #2 - nodes to hyperedge
        #print(f"121212 Ent embed = {ent_embed}")
        msg_type = 2
        propagated_edge_msg2 = self.propagate(hyperedge_index,size=(num_nodes,num_edges),x=(ent_embed,edge_embed),msg_type=msg_type, norm=norm_node_to_edge, qual_edge_embed =None, weight=self.w_nodes_to_edges)
        #Shold we check if any of the following 3 components is zero before doing 1/3?? And if anything is zero should we do a 1/2?
        #updated_edge_embed = edge_embed* (1/3) + propagated_edge_msg1* (1/3) + propagated_edge_msg2* (1/3)
        updated_edge_embed = edge_embed* self.w_combine_edge_embed[0] + propagated_edge_msg1* self.w_combine_edge_embed[1] + propagated_edge_msg2* self.w_combine_edge_embed[2]
        print(f"In COmbine edge embeds ={self.w_combine_edge_embed}")
        print(f"AFter 2nd propagation shape={updated_edge_embed.shape} hyperedge_index.flip([0]) shape")
        #3 - hyperedge to nodes
        msg_type = 3     
        propagated_ent_msg1 = self.propagate(hyperedge_index.flip([0]),size=(num_edges,num_nodes), x=(updated_edge_embed,ent_embed), msg_type=msg_type,norm=norm_edge_to_node,qual_edge_embed = None,weight = self.w_edges_to_nodes)
        #4  hyper edge to quals
        msg_type = 4
        #the weight used for this propgn has to be inverse of W_quals
        propagated_ent_msg2 = self.propagate(qual_edge_index.flip([0]),size=(num_edges,num_nodes), x=(updated_edge_embed,ent_embed), msg_type=msg_type,norm=norm_edge_to_qual,qual_edge_embed=qual_edge_embed,weight = self.w_edges_to_quals)

       #The node updated node embeddings are a sum of self, aggregated neighbours -both primary edge and qual edge
        #do we need to divide the value by 3 or so in order to normailize? 
        #updated_ent_embed = ent_embed * (1/3) + propagated_ent_msg1 * (1/3)+ propagated_ent_msg2 * (1/3)
        updated_ent_embed = ent_embed * self.w_combine_ent_embed[0] + propagated_ent_msg1 * self.w_combine_ent_embed[1] + propagated_ent_msg2 * self.w_combine_ent_embed[2] 
        print(f"In Combine ent embeds ={self.w_combine_edge_embed}")

        #In case of hyper  edges, the relation embeddings have to be updated by propagation from the 
        # neighboring(or member nodes). Need to do it here or Some where else???
        #updated_rel_embed = torch.matmul(rel_embed, self.w_rel) # what transformation is to be applied??
        updated_rel_embed = scatter(updated_edge_embed, edge_type,dim=0, dim_size=num_rels, reduce='mean')
        print(f"Returning from GNN layer  ent_embed device={updated_ent_embed.device} rel embed device = {updated_rel_embed.device}")
        #updated_ent_embed = self.ent_bn(updated_ent_embed)
        #updated_rel_embed = self.rel_bn(updated_rel_embed)

        return updated_ent_embed, updated_rel_embed
    
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
    
    
    def update_hyperedge_with_quals(self,ent_embed,rel_embed,edge_type,edge_qual_details):
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
    def combine_qual_node_with_qual_edge(self,qual_node_embed,qual_edge_embed):
        if self.config['QUAL_TRANSFORM'] == 'corr':
            trans_embed = ccorr(qual_node_embed, qual_edge_embed)
        elif self.config['QUAL_TRANSFORM'] == 'sub':
            trans_embed = qual_node_embed - qual_edge_embed
        elif self.config['QUAL_TRANSFORM'] == 'mult':
            trans_embed = qual_node_embed * qual_edge_embed
        elif self.config['QUAL_TRANSFORM'] == 'rotate':
            trans_embed = rotate(qual_node_embed, qual_edge_embed)
        else:
            raise NotImplementedError
        return trans_embed 
    def combine_hyperedge_with_qual_edge(self,hyper_edge_embed,qual_edge_embed):
        if self.config['QUAL_TRANSFORM'] == 'corr':
            trans_embed = ccorr(hyper_edge_embed, qual_edge_embed)
        elif self.config['QUAL_TRANSFORM'] == 'sub':
            trans_embed = hyper_edge_embed - qual_edge_embed
        elif self.config['QUAL_TRANSFORM'] == 'mult':
            trans_embed = hyper_edge_embed * qual_edge_embed
        elif self.config['QUAL_TRANSFORM'] == 'rotate':
            trans_embed = rotate(hyper_edge_embed, qual_edge_embed)
        else:
            raise NotImplementedError
        return trans_embed 

    def message(self, edge_index,size,x,x_j,x_i,msg_type,norm_i,qual_edge_embed=None, weight=None):
        #norm_i=[]
        #print(f">>>>>>>>Msg type={msg_type} norm_i={norm_i} x shape={x[0].shape, x[1].shape} x_j={x_j.shape} x_i={x_i.shape} \n x_i={x_i} \n x_j ={x_j}")
        #Hyper edges with qualifiers - Hyperedge Hyperrelational graphs
        #1)all qualifiers of an edge are combined
        # 2) update hyperedge embeddings with qualifiers embedding and the member node embeddings
        # 3)The node embeddings are updated using the neighbouring nodes and hyper edge emebddings

        #weight = getattr(self, 'w_{}'.format(mode))
        #Update the rel_embed using the qualifier pairs and then prepare the edge msg
        #using the x_j and the rel_embed
        #x_j , is the source node which sends the message
        #x_i is the dst node that receives the msg 
        #if msg_type=2
            #.the source node is a node in the hyper edge 
            # dst node is the hyperedge itself
            #We need to send it to the hyperedge , where it will be aggregated. each hyperedge gets messages
            # from all the nodes which are members of it to the member nodes.
        #if msg_type=3
            #.the source node is hyperedge itself
            # dst node is the member node  in the hyper edge 
            #propagation happens in the other direction, where we send 
            # the hyperedge features to the member nodes. Each member node gets message from 
            # all the hyperedges of which it is a member
        
     
         #So where do we use the relation embedding?
         #when a hyperedge receives messsages from its member nodes and aggregates, may be 
         #we can include the relation embedding in the aggregation with some weightage/attention
        if msg_type==1:
            print(f"Qual to hyperedge Msg type={msg_type}")
            msg = self.combine_qual_node_with_qual_edge(x_j,qual_edge_embed) 
        elif msg_type==2: # from nodes to edge    
            #x[0] is ent_embed, x[1] is edge embed
            #X_j is the ent_embed 
            msg = x_j 
        elif msg_type==3: #from hyperedges to nodes
            print(f"Hyperedge to nodes Msg type={msg_type}")
            msg = x_j
        elif msg_type==4: #from hyperedges to qual nodes
            msg = self.combine_hyperedge_with_qual_edge(x_j,qual_edge_embed)
      
       
        #out = norm_i.view(-1, 1,1) * x_j.view(-1, H, F)

        out = torch.einsum('ij,jk->ik', msg, weight)
        #out = msg
        #out = x_j
        #out is the matrix containing all the message rows corresponding to x_j for all the edges
        #in the edge_index. Now these out messages will be aggregated for all the x_i (here the hyperedges)
        #using the aggregate function(default is "add") and the x_i is updated using the update function(default is none)

        #Norm is the edge degree in first call and node degree in second call of propagate.
        # Hence multiplying out with norm_i normailzes out by the degree of of edge/node respectively
        #in the first and second call
        return out if norm_i is None else out * norm_i.view(-1, 1)
    def update(self, aggr_out,x):
        #Update is for the destination embedding matrix. Size is that of dst tensor
        #Not based on the number of edges or the messages passed
        #Any weight to be used for combining the self embedding and the neighbours embedding??
        #updated_embed = (aggr_out + x[1])*0.5
        #if aggr_ot is 0 for any particular embedding index, do we need to divide by 2?
        #print(f"@@@@Aggre = {aggr_out}")
        #print(f"@@@@Updated = {updated_embed}")

        return F.relu(aggr_out)
