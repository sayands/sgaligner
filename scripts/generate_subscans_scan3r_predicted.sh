# # Generate Scene Graphs from 3DSSG
# python preprocessing/scan3r/run_genseg.py --config configs/scan3r/scan3r_predicted.yaml --split train --graph_slam_exe ../SceneGraphFusion/bin/exe_GraphSLAM --weights_dir ../weights/traced
# python preprocessing/scan3r/run_genseg.py --config configs/scan3r/scan3r_predicted.yaml --split val --graph_slam_exe ../SceneGraphFusion/bin/exe_GraphSLAM --weights_dir ../weights/traced

# Develop objects + relationships from generated scene graphs
# python preprocessing/scan3r/gen_data_scan3r.py --config configs/scan3r/scan3r_predicted.yaml

# # Generate sub-scans
# python preprocessing/scan3r/generate_subscans.py --config configs/scan3r/scan3r_predicted.yaml --split train
# python preprocessing/scan3r/generate_subscans.py --config configs/scan3r/scan3r_predicted.yaml --split val

# Preprocess data for the framework
python preprocessing/scan3r/preprocess.py --config configs/scan3r/scan3r_predicted.yaml --split train  
python preprocessing/scan3r/preprocess.py --config configs/scan3r/scan3r_predicted.yaml --split val