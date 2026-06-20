import os
import random
import numpy as np
import torch

from torch_geometric.datasets import Planetoid


##Do we really need numpy, if we have tensors in torch?``


class InputDataManager():
    def __init__(self,args):
        self.name = args.dataset
        self.dir = os.path.join("data", self.name)
        self.device = args.device
        # THIS NEEDS TO STAY 6
        self.max_arity = args.max_arity
        self.max_q_pairs = args.max_q_pairs

        # id zero means no entity. Entity ids start from 1.
        self.ent2id = {"":0}
        self.rel2id = {"":0}
        # The unified data structure  for holding both types of complex facts - 
        # hyper-relational and n-ary
        #
        # Each fact will be of the form {"PT":[r, e1,....en], "QP":[[qr1,qe1],[qr2,qe2]....[qrm, qem]]}
        # Where n is the maximum arity of n-ary relations and m is the max number of qualifier pairs in the fact
        # Now the dataset will consist of subsets "train", "valid" and "test"
        self.data = {}
        print("Loading the dataset {} ....".format(self.name))
        
        if self.name in ["cora"]:
            self.data["graph_data"], self.data["node_features"], self.data["labels"], self.data["train_mask"], self.data["test_mask"], self.data["val_mask"] = self.get_cora_dataset()
        else:
            self.data["train"] = self.read(os.path.join(self.dir, "train.txt"))
            self.data["graph_data"] = self.data["train"] #graph data is same as the training data in this case
            print("Read Train the dataset {} ....".format(self.data["train"].shape))
            # Load the test data
            self.data["test"] = self.read(os.path.join(self.dir, "test.txt"))
            # Read the test files by arity, if they exist
            # If they do, then test output will be displayed by arity
            """ for i in range(2,self.max_arity+1):
                test_arity = "test_{}".format(i)
                file_path = os.path.join(self.dir, "test_{}.txt".format(i))
                self.data[test_arity] = self.read_test(file_path) """

            self.data["valid"] = self.read(os.path.join(self.dir, "valid.txt"))


        #Convert graph data into a representation suitable for GNN/GCN propagation
        
        graph_representation = {}
        graph_representation["edge_index"], graph_representation["edge_type"], graph_representation["qual_details"]  = self.convert_to_graph_representation_for_msg_passing(self.data["graph_data"])
        self.graph_representation = graph_representation
        self.num_entities = len(self.ent2id)
        self.num_relations = len(self.rel2id)
        self.num_tuples = len(self.data["graph_data"])
        #print(f"Enities2id= {self.ent2id} Rel2id={self.rel2id}")
        print(f"Num tuples = {self.num_tuples}  NUm Entities={self.num_entities}  num_relations={self.num_relations}")

    def convert_to_graph_representation_for_msg_passing(self,data):
        primary_tuples=data[:,:self.max_arity]
        qual_pairs = data[:,self.max_arity+1:]
        print(f"@@@@@@@Convert to graph data shape={data.shape} pr_tuples shape={primary_tuples.shape} qual_pairs={qual_pairs.shape}")

        #the len of both primary_tuples and qual_pairs should be equal
        num_tuples = len(primary_tuples)
        # We need to add inverse tuples for each tuple? 
        # How do we define inverse relation in case of hyper-edges?
        #
        #np_hyperedge_index, np_hyperedge_type = np.zeros((2, num_tuples * 2), dtype='int32'), np.zeros((num_tuples * 2), dtype='int32')
        #np_hyperedge_index, np_hyperedge_type = np.zeros((2, num_tuples), dtype='int32'), np.zeros((num_tuples), dtype='int32')
        #np_hyperedge_index, np_hyperedge_type = np.empty((2, 0)), np.empty((0))
        np_hyperedge_index, np_hyperedge_type = [[],[]], []
        qualifier_rel = []
        qualifier_ent = []
        qualifier_edge = []

       

        # Add actual data
        for i, pr_tuple in enumerate(primary_tuples):
            np_hyperedge_type.append(pr_tuple[0])
            #Need to loop only till the entity is 0 or if we know th arity value, use that
            for j,ent in enumerate(pr_tuple[1:]):
                if ent == 0:
                    break
                np_hyperedge_index[0].append(ent)
                np_hyperedge_index[1].append(i)
                print(f"In for loop i={i} pr_tuple={pr_tuple} index 0 ={np_hyperedge_index[0][-1]} index 1={np_hyperedge_index[1][-1]} ")


            qual_rel = np.array(qual_pairs[i][::2])
            qual_ent = np.array(qual_pairs[i][1::2])
            non_zero_rels = qual_rel[np.nonzero(qual_rel)]
            non_zero_ents = qual_ent[np.nonzero(qual_ent)]
            if len(non_zero_ents) > 0:
                #print(f"Non zero quals num={len(non_zero_ents)}")
                for j in range(non_zero_ents.shape[0]):
                    qualifier_rel.append(non_zero_rels[j])
                    qualifier_ent.append(non_zero_ents[j])
                    qualifier_edge.append(i)

        np_qual_details = np.stack((qualifier_rel, qualifier_ent, qualifier_edge), axis=0)

        hyperedge_index = torch.tensor(np.array(np_hyperedge_index), dtype=torch.long, device=self.device)
        hyperedge_type = torch.tensor(np.array(np_hyperedge_type), dtype=torch.long, device=self.device)
        qual_details = torch.tensor(np_qual_details, dtype=torch.long, device=self.device)
        print(f"Edge_index shape={hyperedge_index.shape} Edge_index ={hyperedge_index} Edge_Type={hyperedge_type}")
        return hyperedge_index, hyperedge_type, qual_details

        

    def read(self, file_path):
            if not os.path.exists(file_path):
                print("*** {} not found. Skipping. ***".format(file_path))
                return ()
            """ with open(file_path, "r") as f:
                lines = f.readlines() """
            lines = open(file_path, "r").read().splitlines()
            #We need to create hyperegde indexes that can be used  in the message passing
            #But unlike HyperGNN that handles only single relational graphs, we need to have a mechanism to 
            #factor in the relation type of each hyperedge
            #Then we need qual pairs also to be linked to the hyper edges
            """  primary_tuples = [] #array of all the primary tuples
            qual_pairs = []  #array of all corresponding qual pairs
            #qual_pairs =
            #qual_pair_indices = 
            data = {}
            # Shuffle the train set
            np.random.shuffle(lines)
            for i, line in enumerate(lines):                
                pr_tuple_id, q_pairs_id = self.tuple2ids(i,line)
                primary_tuples.append(pr_tuple_id)
                qual_pairs.append(q_pairs_id)
            
            data = {"primary_tuples":primary_tuples, "qual_pairs":qual_pairs} """
            statements = []
            np.random.shuffle(lines)
            for i, line in enumerate(lines):                
                pr_tuple_id, q_pairs_id = self.tuple2ids(i,line)
                stmt = np.concatenate((pr_tuple_id,q_pairs_id),axis=0)
                statements.append(stmt)
            return np.array(statements,dtype=np.int64)
    
    """ def read_test(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = np.zeros((len(lines),  self.max_arity + 1))
        for i, line in enumerate(lines):
            splitted = line.strip().split("\t")[1:]
            tuples[i] = self.tuple2ids(splitted)
        return tuples """
    
    def tuple2ids(self, line_num,line):
        #print(f"Line num={line_num} for Tuple 2 Ids {line}")
        #Validate the input line -- 1 reln and max_arity entities for primary tuple
        #-- equal number of relns and entities for q pairs, max pairs to be max_q_pairs
        pr_tuple = tuple((line.partition("<<")[2].partition(">>")[0]).split(','))
        q_pairs =  tuple((line.partition(">>")[2]).split(','))
        #print(f"Partitioned string pr_tuple= {pr_tuple} q_pairs={q_pairs} ")
        #!!!!Below line only for testing without qualifiers
        #!!!!q_pairs = tuple()
        pr_tuple_id = np.zeros(self.max_arity + 1)
        for ind,t in enumerate(pr_tuple):
            #print(f"enumerate {ind}<>{t}")
            if ind == 0:
                pr_tuple_id[ind] = self.get_rel_id(t)
            else:
                pr_tuple_id[ind] = self.get_ent_id(t)
        
        q_pairs_id = np.zeros(self.max_q_pairs * 2)
        for ind,t in enumerate(q_pairs):
            #print(f"enumerate {ind}<>{t}")
            if ind < self.max_q_pairs * 2:
                if ind % 2 == 0:
                    q_pairs_id[ind] = self.get_rel_id(t)
                else:
                    q_pairs_id[ind] = self.get_ent_id(t)
        
        #print(f">>>>>Tuple 2 Ids output pr_tuple_id {pr_tuple_id} \n qual_pairs_id = {q_pairs_id}")

        return pr_tuple_id,q_pairs_id

    def get_ent_id(self, ent):
        print(f"Getting ent id for {ent}")
        ent = str(ent.item()) 
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id) 
            print(f"Assigned id {self.ent2id[ent]} to entity {ent}")
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]


    def get_cora_dataset(self):

        if not os.path.exists('cora_graph.pt'):
            print("Cora dataset not found. Downloading...")
            dataset = Planetoid(root='/tmp/Cora', name='Cora')
            torch.save(dataset, 'cora_graph.pt')
            print("Dataset successfully saved as cora_graph.pt")

        dataset = torch.load('cora_graph.pt')
        data = dataset[0]  # Cora contains exactly one graph object
        # Inspect the dataset properties
        print(f"Number of graphs: {len(dataset)}")
        print(f"Data structure: {data}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Number of node features: {data.num_node_features}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Edge index : {data.edge_index}")
        statements = []
       
        #np.random.shuffle(data.edge_index)
        
        pr_tuple_id = np.zeros(self.max_arity + 1)
    
        self.get_rel_id("cora_edge")

        for i in range(data.edge_index.size(1)):  
            pr_tuple_id[0] = 1  # Assuming a single relation type for Cora
            pr_tuple_id[1] = self.get_ent_id(data.edge_index[0][i])
            pr_tuple_id[2] = self.get_ent_id(data.edge_index[1][i])
            stmt = pr_tuple_id
            statements.append(stmt) #the complete graph is used for training 

        
        #print(f"@@@@@@@Convert to graph data shape={statements} ")
        return np.array(statements,dtype=np.int64),data.x, data.y, data.train_mask, data.test_mask, data.val_mask




