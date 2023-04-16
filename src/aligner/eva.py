import torch
import torch.nn as nn
import torch.nn.functional as F

from aligner.networks.gat import MultiGCN
from aligner.networks.pointnet import PointNetfeat
from aligner.sg_aligner import MultiModalFusion

class EVA(nn.Module):
    def __init__(self, modules, rel_dim, attr_dim, n_units=[3, 200, 400], emb_dim = 100, pt_out_dim = 256, dropout = 0.0, attn_dropout = 0.0, instance_norm = False):
        super(EVA, self).__init__()
        self.modules = modules
        self.pt_out_dim = pt_out_dim
        self.rel_dim = rel_dim
        self.emb_dim = emb_dim
        self.attr_dim =  attr_dim
        self.n_units = n_units
        
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.instance_norm = instance_norm
        self.inner_view_num = len(self.modules) # Point Net + Structure Encoder + Meta Encoders

        self.meta_embedding_rel = nn.Linear(self.rel_dim, self.emb_dim)
        self.meta_embedding_attr = nn.Linear(self.attr_dim, self.emb_dim)
        
        self.object_encoder = PointNetfeat(global_feat=True, batch_norm=True, point_size=3, input_transform=False, feature_transform=False, out_size=200)
        
        self.structure_encoder = MultiGCN(n_units=self.n_units, dropout=self.dropout)
    
        self.fusion = MultiModalFusion(modal_num=self.inner_view_num, with_weight=1)
    
    def forward(self, data_dict):
        tot_object_points = data_dict['tot_obj_pts'].permute(0, 2, 1)
        tot_bow_vec_object_attr_feats = data_dict['tot_bow_vec_object_attr_feats'].float() 
        tot_bow_vec_object_edge_feats = data_dict['tot_bow_vec_object_edge_feats'].float()
        tot_rel_pose = data_dict['tot_rel_pose'].float()
        
        start_object_idx = 0
        start_edge_idx = 0
        batch_size = data_dict['batch_size']
        embs = {}
        
        for module in self.modules:
            if module == 'gcn':
                structure_embed = None
                for idx in range(batch_size):
                    src_object_count = data_dict['graph_per_obj_count'][idx][0]
                    ref_object_count = data_dict['graph_per_obj_count'][idx][1]

                    src_edges_count = data_dict['graph_per_edge_count'][idx][0]
                    ref_edges_count = data_dict['graph_per_edge_count'][idx][1]
                    
                    src_objects_rel_pose = tot_rel_pose[start_object_idx : start_object_idx + src_object_count]
                    start_object_idx += src_object_count

                    ref_objects_rel_pose = tot_rel_pose[start_object_idx : start_object_idx + ref_object_count]
                    start_object_idx += ref_object_count

                    src_edges = torch.transpose(data_dict['edges'][start_edge_idx : start_edge_idx + src_edges_count], 0, 1).to(torch.int32)
                    start_edge_idx += src_edges_count

                    ref_edges = torch.transpose(data_dict['edges'][start_edge_idx : start_edge_idx + ref_edges_count], 0, 1).to(torch.int32)
                    start_edge_idx += ref_edges_count
                    
                    src_structure_embedding = self.structure_encoder(src_objects_rel_pose, src_edges)
                    ref_structure_embedding = self.structure_encoder(ref_objects_rel_pose, ref_edges)
                    
                    structure_embed = torch.cat([src_structure_embedding, ref_structure_embedding]) if structure_embed is None else \
                                   torch.cat([structure_embed, src_structure_embedding, ref_structure_embedding]) 

                emb = structure_embed #self.structure_embedding(structure_embed)
            
            elif module == 'point':
                emb = self.object_encoder(tot_object_points)

            elif module == 'rel':
                emb = self.meta_embedding_rel(tot_bow_vec_object_edge_feats)
            
            elif module == 'attr':
                emb = self.meta_embedding_attr(tot_bow_vec_object_attr_feats)
            
            else:
                raise NotImplementedError
            
            embs[module] = emb

        if len(self.modules) > 1:
            all_embs = []
            for module in self.modules:
                all_embs.append(embs[module])
            
            joint_emb = self.fusion(all_embs)
            embs['joint'] = joint_emb
    
        return embs