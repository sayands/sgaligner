
def compute_mean_reciprocal_rank(rank_list, e1i_idxs, e2i_idxs, mrr_arr):
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = rank_list[e1i_idx]
        e1_idx_rank_list = list(e1_idx_rank_list)
        e1_idx_rank_list.remove(e1i_idx)
        rank = e1_idx_rank_list.index(e2i_idxs[idx]) + 1
        mrr_arr.append(1.0 / rank)
    
    return mrr_arr

def compute_hits_k(rank_list, e1i_idxs, e2i_idxs, k=1):
    correct, total = 0, 0
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = rank_list[e1i_idx]
        e1_idx_rank_list = list(e1_idx_rank_list) 
        e1_idx_rank_list.remove(e1i_idx)

        e1_idx_rank_list_k = e1_idx_rank_list[:k]

        if e2i_idxs[idx] in e1_idx_rank_list_k: correct += 1
    
    total = e1i_idxs.shape[0]

    return correct, total

def compute_node_corrs(rank_list, e1i_idxs, src_objects_count, k=1):
    node_corrs = []
    for idx, e1i_idx in enumerate(e1i_idxs):
        e1_idx_rank_list = rank_list[e1i_idx]
        e1_idx_rank_list = list(e1_idx_rank_list) 
        e1_idx_rank_list.remove(e1i_idx)

        for k_idx in range(0, k):
            if e1_idx_rank_list[k_idx] < src_objects_count: continue # Ignore self-graph matches
            node_corrs.append((e1i_idx, e1_idx_rank_list[k_idx]))
    
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





