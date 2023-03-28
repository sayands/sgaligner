import os.path as osp

def get_class_2_idx(file_name):
    class_2_idx = {}
    index = 0
    class_file_name = osp.join(file_name)

    with open(class_file_name) as f:
        lines = f.readlines()
        for line in lines:
            class_name = line.split('\t')[1]
            class_2_idx[class_name] = index
            index += 1
    
    return class_2_idx