# # Generate the subscenes
python preprocessing/scan3r/generate_subscans.py --config configs/scan3r/scan3r_ground_truth.yaml --split train
python preprocessing/scan3r/generate_subscans.py --config configs/scan3r/scan3r_ground_truth.yaml --split val

# Preprocess data for the framework
python preprocessing/scan3r/preprocess.py --config configs/scan3r/scan3r_ground_truth.yaml --split train
python preprocessing/scan3r/preprocess.py --config configs/scan3r/scan3r_ground_truth.yaml --split val

python preprocessing/gen_all_pairs_fileset.py
cd src
python trainers/trainval_sgaligner.py --config ../configs/scan3r/scan3r_ground_truth.yaml

python inference/sgaligner/inference_align_reg.py --config ../configs/scan3r/scan3r_ground_truth.yaml --snapshot ../output/Scan3R/sgaligner/pct_gat_rel_attr/snapshots/best_snapshot.pth.tar --reg_snapshot /drive/pretrained-models/geotransformer-3dmatch.pth.tar