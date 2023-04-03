import torch
from torch import nn
import torch.nn.functional as F

def calculate_prob_dist(e1i, e2i, e1j, e2j, temp):
    deltaM_e1i_e2i = torch.exp(torch.matmul(e1i, torch.transpose(e2i, 0, 1)) / temp)
    deltaM_e1i_e1j = torch.exp(torch.matmul(e1i, torch.transpose(e1j, 0, 1)) / temp)
    deltaM_e1i_e2j = torch.exp(torch.matmul(e1i, torch.transpose(e2j, 0, 1)) / temp)

    deltaM_e1i_e2i_e1j = deltaM_e1i_e2i / (deltaM_e1i_e1j.sum() + 1e-9)
    deltaM_e1i_e2i_e2j = deltaM_e1i_e2i / (deltaM_e1i_e2j.sum() + 1e-9)
    q_e1i_e2i_inverse = 1.0 + 1.0 / (deltaM_e1i_e2i_e1j + 1e-9) + 1.0 / (deltaM_e1i_e2i_e2j + 1e-9)
    q_e1i_e2i = 1.0 / (q_e1i_e2i_inverse + 1e-9)

    return q_e1i_e2i

class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss

class ICLLoss(nn.Module):
    def __init__(self, device, temperature=0.05, alpha = 0.5):
        super(ICLLoss, self).__init__()
        self.temp = 0.1 #temperature
        self.alpha = alpha
        self.device = device
    
    def forward(self, emb, data_dict):
        emb = F.normalize(emb, dim=1)
        e1i = emb[data_dict['e1i']]
        e2i = emb[data_dict['e2i']]
        e1j = emb[data_dict['e1j']]
        e2j = emb[data_dict['e2j']]

        qm_e1i_e2i = calculate_prob_dist(e1i, e2i, e1j, e2j, self.temp)
        qm_e2i_e1i = calculate_prob_dist(e2i, e1i, e2j, e1j, self.temp)

        lossA = qm_e1i_e2i
        lossB = qm_e2i_e1i

        loss = self.alpha * lossA + (1-self.alpha) * lossB
        loss = -torch.log(loss).mean()
        return loss

class IALLoss(nn.Module):
    def __init__(self, device, temperature=0.05, alpha = 0.5):
        super(IALLoss, self).__init__()
        self.temp = 1.0 #temperature
        self.alpha = alpha
        self.device = device
        self.zoom = 0.1

    def forward(self, src_emb, ref_emb, data_dict):
        '''
        srcEmb : joint embedding
        ref_emb : modal embedding
        '''

        src_emb = F.normalize(src_emb, dim=1)
        ref_emb = F.normalize(ref_emb, dim=1)

        o_e1i = src_emb[data_dict['e1i']]
        o_e2i = src_emb[data_dict['e2i']]
        o_e1j = src_emb[data_dict['e1j']]
        o_e2j = src_emb[data_dict['e2j']]

        qo_e1i_e2i = calculate_prob_dist(o_e1i, o_e2i, o_e1j, o_e2j, self.temp)
        qo_e2i_e1i = calculate_prob_dist(o_e2i, o_e1i, o_e2j, o_e1j, self.temp)

        m_e1i = ref_emb[data_dict['e1i']]
        m_e2i = ref_emb[data_dict['e2i']]
        m_e1j = ref_emb[data_dict['e1j']]
        m_e2j = ref_emb[data_dict['e2j']]

        qm_e1i_e2i = calculate_prob_dist(m_e1i, m_e2i, m_e1j, m_e2j, self.temp)
        qm_e2i_e1i = calculate_prob_dist(m_e2i, m_e1i, m_e2j, m_e1j, self.temp)
        
        klLoss = nn.KLDivLoss(size_average=False, reduction="sum", log_target=True)
        loss_a = klLoss(qm_e1i_e2i.log(), qo_e1i_e2i).mean()
        loss_b = klLoss(qm_e2i_e1i.log(), qo_e2i_e1i).mean()

        loss = self.zoom * (self.alpha * loss_a + (1-self.alpha) * loss_b)
        return loss


class OverallLoss(nn.Module):
    def __init__(self, ial_loss_layer, icl_loss_layer, device, metadata):
        super(OverallLoss, self).__init__()
        
        self.zoom = metadata['zoom']
        self.device = device
        self.modules = metadata['modules']
        self.weight_align_loss = metadata['wt_align_loss']
        self.weight_contrastive_loss = metadata['wt_contrastive_loss']

        self.align_loss = IALLoss(device)
        self.contrastive_loss = ICLLoss(self.device)
        self.align_multi_loss_layer = ial_loss_layer
        self.contrastive_multi_loss_layer = icl_loss_layer

    def forward(self, output_dict, data_dict):
        # alignment loss
        if len(self.modules) > 1:
            align_losses = []
            for module in self.modules:
                loss = self.align_loss(output_dict[module], output_dict['joint'], data_dict)
                align_losses.append(loss)
            
            total_align_loss = self.align_multi_loss_layer(align_losses) * self.zoom
        
        # contrastive loss - unimodal
        constrastive_losses_unimodal = []
        for module in self.modules:
            loss = self.contrastive_loss(output_dict[module], data_dict)
            constrastive_losses_unimodal.append(loss)
        
        if len(self.modules) > 1:
            constrastive_loss_unimodal = self.contrastive_multi_loss_layer(constrastive_losses_unimodal)
        else:
            constrastive_loss_unimodal = constrastive_losses_unimodal[0]

        # constrastive loss - multi-modal
        constastive_loss_multimodal = self.contrastive_loss(output_dict['joint'], data_dict)

        loss = total_align_loss + constrastive_loss_unimodal + constastive_loss_multimodal
        return {
            'loss': loss,
            'icl_loss_unimodal': constrastive_loss_unimodal,
            'icl_loss_multimodal' : constastive_loss_multimodal,
            'ial_loss': total_align_loss,
        }