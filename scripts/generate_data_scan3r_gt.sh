# Generate the subscenes
python data-preprocessing/gen_subscene_scan3r.py --split train
python data-preprocessing/gen_subscene_scan3r.py --split val

# Preprocess data for the framework
python data-preprocessing/preprocess_scan3r.py --split train
python data-preprocessing/preprocess_scan3r.py --split val