import os.path as osp

SCAN3R_ORIG_DIR = '/drive/datasets/Scan3R'
SCAN3R_SUBSCENES_DIR = '/drive/dumps/sgaligner/subscans/Scan3R'

LABEL_FILE_NAME_GT = 'labels.instances.align.annotated.v2.ply'
LABEL_MAPPING_FILE = osp.join(SCAN3R_ORIG_DIR, 'files', '3RScan.v2 Semantic Classes - Mapping.csv')
CLASS160_FILE = osp.join(SCAN3R_ORIG_DIR, 'files', 'classes160.txt')

OBJ_ATTR_FILENAME = osp.join(SCAN3R_ORIG_DIR, 'files/obj_attr.pkl')