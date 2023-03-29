import json
import os
import os.path as osp
import pickle

def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)

def assert_dir(path):
    assert osp.exists(path)

def load_pkl_data(filename):
    with open(filename, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict

def write_pkl_data(data_dict, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_json(filename):
    file = open(filename)
    data = json.load(file)
    file.close()
    return data

def write_json(data_dict, filename):
    json_obj = json.dumps(data_dict, indent=4)
 
    with open(filename, "w") as outfile:
        outfile.write(json_obj)