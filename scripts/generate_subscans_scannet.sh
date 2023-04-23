# Generate Scene Graphs from 3DSSG
python preprocessing/scannet/run_genseg.py --config configs/scannet/scannet.yaml --split train --graph_slam_exe ../SceneGraphFusion/bin/exe_GraphSLAM --weights_dir ../weights/traced

# Develop objects + relationships from generated scene graphs using correspondences
python preprocessing/scannet/gen_data_scannet.py --config configs/scannet/scannet.yaml --split train
python preprocessing/scannet/gen_data_scannet.py --config configs/scannet/scannet.yaml --split val
