from .scan3r import Scan3RDataset

def get_dataset(dataset_name):
    if dataset_name == 'Scan3R':
        return Scan3RDataset
    else:
        raise NotImplementedError