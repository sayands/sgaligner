# Generate the subscenes
python preprocessing/scan3r/generate_subscans.py --config configs/scan3r/scan3r_ground_truth.yaml --split train
python preprocessing/scan3r/generate_subscans.py --config configs/scan3r/scan3r_ground_truth.yaml --split val

# Preprocess data for the framework
python preprocessing/scan3r/preprocess.py --config configs/scan3r/scan3r_ground_truth.yaml --split train
python preprocessing/scan3r/preprocess.py --config configs/scan3r/scan3r_ground_truth.yaml --split val