import numpy as np 

def compute_mean_reciprocal_rank(rank_list, e1i_idxs, e2i_idxs, mrr_arr):
    rank_list = rank_list.detach().cpu().numpy()
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = list(rank_list[e1i_idx])
        e1_idx_rank_list.remove(e1i_idx)
        rank = e1_idx_rank_list.index(e2i_idxs[idx]) + 1
        mrr_arr.append(1.0 / rank)
    
    return mrr_arr

def compute_hits_k(rank_list, e1i_idxs, e2i_idxs, k=1):
    rank_list = rank_list.detach().cpu().numpy()
    correct, total = 0, 0
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = list(rank_list[e1i_idx])
        e1_idx_rank_list.remove(e1i_idx)
        e1_idx_rank_list_k = e1_idx_rank_list[:k]

        if e2i_idxs[idx] in e1_idx_rank_list_k: correct += 1
    
    total = e1i_idxs.shape[0]

    return correct, total

def compute_sgar(sim, rank_list, e1i_idxs, e2i_idxs, modes):
    pair_data = { 'pred_matches' : [] , 'gt_matches' : [], 'sim' : []}
    rank_list = rank_list.detach().cpu().numpy()
    sim = sim.detach().cpu().numpy()
    
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1i_idx_rank_list = list(rank_list[e1i_idx])
        e1i_idx_rank_list.remove(e1i_idx)

        pair_data['pred_matches'].append(e1i_idx_rank_list[0])
        pair_data['sim'].append(sim[e1i_idx][e1i_idx_rank_list[0]])
        pair_data['gt_matches'].append(e2i_idxs[idx])
    
    sim_argsorted = np.argsort(pair_data['sim'])
    sgar_vals = {}

    for mode in modes:
        if mode == '2'    : sim_argsorted_mode = sim_argsorted[: 2]
        elif mode == '50' : sim_argsorted_mode = sim_argsorted[: len(sim_argsorted) // 2]
        else : sim_argsorted_mode = sim_argsorted

        mis_res = False
        for idx in sim_argsorted_mode:
            if pair_data['pred_matches'][idx] != pair_data['gt_matches'][idx]:
                mis_res = True
                break
    
        if mis_res: sgar_vals[mode] = 0.0
        else      : sgar_vals[mode] = 1.0
    
    return sgar_vals

def compute_node_corrs(rank_list, src_objects_count, k=1):
    rank_list = rank_list.detach().cpu().numpy()
    node_corrs = []
    for idx in range(src_objects_count):
        entity_rank_list = list(rank_list[idx])
        entity_rank_list.remove(idx)
        entity_rank_list_k = entity_rank_list[:k]

        for k_idx in range(0, k):
            if entity_rank_list_k[k_idx] < src_objects_count: continue
            node_corrs.append((idx, entity_rank_list_k[k_idx]))
    return node_corrs

def get_node_corrs_objects_ids(node_corrs, objects_ids, batch_offset):
    node_corrs_objects_ids = []

    for node_corr in node_corrs:
        node_corrs_objects_ids.append((objects_ids[node_corr[0] + batch_offset], objects_ids[node_corr[1] + batch_offset]))
    return node_corrs_objects_ids

def compute_alignment_score(rank_list, src_objects_count, ref_objects_count):
    aligned_obj_counts = 0
    for idx in range(src_objects_count):
        e1_rank_list = list(rank_list[idx].detach().cpu().numpy())
        e1_rank_list.remove(idx)
        rank_idx = e1_rank_list[0]

        if rank_idx >= src_objects_count: aligned_obj_counts += 1
    
    alignment_score = aligned_obj_counts / ref_objects_count
    return alignment_score






