import torch
from utils import get_param
import math
from torch.nn import functional as F, Parameter



class HypEDecoder(torch.nn.Module):
    def __init__(self,dataset,config):
        super().__init__()
        self.num_sample =0
        self.num_ent = dataset.num_entities
        self.num_rel = dataset.num_relations
        self.in_channels = config.hype_in_channels
        self.out_channels = config.hype_out_channels
        self.filt_h = config.hype_filt_h
        self.filt_w = config.hype_filt_w
        self.stride = config.hype_stride
        self.hidden_drop_rate = config.hype_hidden_drop
        self.emb_dim = config.emb_dim
        self.max_arity = 6
        rel_emb_dim = self.emb_dim
        print(f"Num entities={dataset.num_entities} Num rels={dataset.num_relations}")
        #self.E = torch.nn.Embedding(d.num_ent(), emb_dim, padding_idx=0)
        #self.R = torch.nn.Embedding(d.num_rel(), rel_emb_dim, padding_idx=0)

        # self.bn0 = torch.nn.BatchNorm2d(self.in_channels) # not used
        # Drop is not reqd as this is only a decoder and drop is meant to act as a regularizer
        #self.inp_drop = torch.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((self.emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, self.emb_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(rel_emb_dim + self.max_arity + 1, fc1_length)
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length)
        self.bias = get_param((self.num_ent), 0) 


    """ 
    The init function is called directlyfrom main() in Hype. Here we don't need this
    def init(self):
        self.E.weight.data[0] = torch.ones(self.emb_dim)
        self.R.weight.data[0] = torch.ones(self.emb_dim)
        xavier_uniform_(self.E.weight.data[1:])
        xavier_uniform_(self.R.weight.data[1:]) """

    def convolve(self,ent_embed, rel_embed, r_idx, e_idx, pos):
        
        e = ent_embed[e_idx].view(-1, 1, 1, self.emb_dim)
        print(">>>>>e dimensions", e.shape)

       
        r = rel_embed[r_idx]
        x = e
        #x = self.inp_drop(x)
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        print(f"x shape {x.shape}")
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, ent_embed, rel_embed, r_idx, e1_idx, e2_idx, e3_idx, e4_idx, e5_idx, ms, bs):
        #print(f"####forward propagation{r_idx.shape}....{e1_idx.shape}")
        r = rel_embed[r_idx]
        e1 = self.convolve(ent_embed, rel_embed,r_idx, e1_idx, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
        e2 = self.convolve(ent_embed, rel_embed,r_idx, e2_idx, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
        e3 = self.convolve(ent_embed, rel_embed,r_idx, e3_idx, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
        e4 = self.convolve(ent_embed, rel_embed,r_idx, e4_idx, 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
        e5 = self.convolve(ent_embed, rel_embed,r_idx, e5_idx, 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)
        #e6 = self.convolve(ent_embed, rel_embed,r_idx, e6_idx, 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)

        pred = e1 * e2 * e3 * e4 * e5 * r
        #pred = self.hidden_drop(pred)

        sim_score = torch.mm(pred, ent_embed.transpose(1, 0))
        sim_score += self.bias.expand_as(sim_score)
        #print(f"####forward propagation{r.shape}..{e1.shape}")
        #x = torch.sum(x, dim=1)
        score = torch.sigmoid(sim_score)
        return score

class DistMultDecoder(torch.nn.Module):
        def __init__(self,dataset,config):
            super().__init__()
            self.num_ent = dataset.num_entities
            self.num_rel = dataset.num_relations
            self.bias = get_param((self.num_ent), 0) 
            #self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

        def forward(self, ent_embed, rel_embed,r_idx, e1_idx, e2_idx, e3_idx, e4_idx,e5_idx,ms, bs):
            #sub_emb = torch.index_select(ent_embed, 0, sub)
            #rel_emb = torch.index_select(rel_embed, 0, rel)
            e1 = ent_embed[e1_idx]* ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e2 = ent_embed[e2_idx]* ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e3 = ent_embed[e3_idx]* ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
            e4 = ent_embed[e4_idx]* ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
            e5 = ent_embed[e5_idx]* ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)

            r = rel_embed[r_idx]

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

            #DIstMult decoder as given in CompGCN
            pred = r * e1 * e2 * e3 * e4 * e5
            sim_score = torch.mm(pred, ent_embed.transpose(1, 0))
            sim_score += self.bias.expand_as(sim_score)

            #In StarE Transformer is used as the decoder for predicting the entity embedding
            #The score generated is as a probability over all the entities
            print(f"^^^^^^^^^^In Predict entity Max={torch.max(pred)} pred= {pred}")
            #score = torch.sigmoid(sim_score)
            score = sim_score
            return score