import os.path as osp

def class_2_idx_scan3r(scan3r_dir):
    class_2_idx = {}
    index = 0
    class_txt_filename = osp.join(scan3r_dir, 'files/classes.txt')

    with open(class_txt_filename) as f:
        lines = f.readlines()
        for line in lines:
            class_name = line.split('\t')[1]
            class_2_idx[class_name] = index
            index += 1
    
    return class_2_idx

def rel_2_idx_scan3r(scan3r_dir):
    rel_2_idx = {}
    index = 0
    rel_txt_filename = osp.join(scan3r_dir, 'files/relationships.txt')

    with open(rel_txt_filename) as f:
        lines = f.readlines()
        for line in lines:
            rel_name = line.split('\n')[0]
            rel_2_idx[rel_name] = index 
            index += 1
    
    return rel_2_idx