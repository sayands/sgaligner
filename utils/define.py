import os.path as osp

DATASET_DIR = '/media/sdebsarkar/extra-hdd/datasets/'
SCAN3R_ORIG_DIR = osp.join(DATASET_DIR, '3RScan')
SCAN3R_SUBSCENES_DIR = osp.join(SCAN3R_ORIG_DIR, 'out')

SCAN3R_PREDICTED_DIR = osp.join(SCAN3R_ORIG_DIR, 'predicted')
SCAN3R_PREDICTED_SUBSCENES_DIR = osp.join(SCAN3R_PREDICTED_DIR, 'out')

LABEL_FILE_NAME_GT = 'labels.instances.align.annotated.v2.ply'
LABEL_FILE_NAME_PRED = 'inseg.ply'

SCANNET_ORIG_DIR = osp.join(DATASET_DIR, 'Scannet')
LABEL_MAPPING_FILE = osp.join(SCAN3R_ORIG_DIR, 'files', '3RScan.v2 Semantic Classes - Mapping.csv')
CLASS160_FILE = osp.join(SCAN3R_ORIG_DIR, 'files', 'classes160.txt')

OBJ_ATTR_FILENAME = osp.join(SCAN3R_SUBSCENES_DIR, 'files/obj_attr.pkl')
NAME_SAME_PART = 'same part'