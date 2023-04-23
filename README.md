## SGAligner : 3D Scene Alignment with Scene Graphs
<div align='center'>

<a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup>1</sup>, <a href="https://miksik.co.uk/">Ondrej Miksik</a><sup>2</sup>, <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>, <a href="https://www.linkedin.com/in/d%C3%A1niel-bar%C3%A1th-3a489092/">Daniel Barath</a><sup>1</sup>, <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft Mixed Reality & AI Labs

</div>

![teaser](https://sayands.github.io/sgaligner/static/images/teaser.png)
SGAligner aligns 3D scene graphs of environments using multi-modal learning and leverage the output for the downstream task of 3D point cloud registration.

## News :newspaper:

* **10. April 2023**: Code released.

## Code Structure :clapper:

```
├── sgaligner
│   ├── data-preprocessing            <- subscan generation + preprocessing
│   ├── configs                       <- configuration files
│   ├── src
│   │   │── aligner                   <- SGAligner modules
│   │   │── datasets                  <- dataloader for 3RScan subscans
│   │   │── engine                    <- trainer classes
│   │   │── GeoTransformer            <- geotransformer submodule for registration
│   │   │── inference                 <- inference files for alignment + downstream applications
│   │   │── trainers                  <- train + validation loop (EVA + SGAligner)
│   │── utils                         <- util functions
│   │── README.md                    
│   │── scripts                       <- bash scripts for data generation + preprocesing + training
│   └── output                        <- folder that stores models and logs
│
```

### Dependencies :memo:

The main dependencies of the project are the following:
```yaml
python: 3.8.15
cuda: 11.6
```
You can set up a conda environment as follows :
```bash
git clone --recurse-submodules -j8 git@github.com:sayands/sgaligner.git
cd sgaligner
conda env create -f req.yml
```

Please follow the submodule for additional installation requirements and setup of [GeoTransformer](https://github.com/sayands/GeoTransformer).

### Downloads :droplet:
The pre-trained model and other meta files are available [here](https://drive.google.com/drive/folders/1elqdbYD5T2686r42lcUHE6SyiFnDsZur?usp=sharing).

### Data + Benchmark :hammer:
After installing the dependencies, we preprocess the datasets and provide the benchmarks. 

#### Subscan Pair Generation - 3RScan + 3DSSG
Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Move all files of 3DSSG to a new ``files/`` directory within Scan3R. The structure should be:

```
├── 3RScan
│   ├── files       <- all 3RScan and 3DSSG meta files (NOT the scan data)  
│   ├── scenes      <- scans
│   └── out         <- Default output directory for generated subscans (created when running pre-processing)
```

Change the absolute paths in ``utils/define.py``.

First, we create sub-scans from each 3RScan scan using the ground truth scene Graphs from the 3DSSG dataset and then calculate the pairwise overlap ratio for the subscans in a scan. Finally, we preprocess the data for our framework. The relevant code can be found in the ``data-preprocessing/`` directory. You can use the following command to generate the subscans.

```bash
bash scripts/generate_data_scan3r_gt.sh
```
> __Note__ To adhere to our evaluation procedure, please do not change the seed value in the files in ``configs/`` directory. 

#### Generating Overlapping and Non-Overlapping Subscan Pairs
To generate overlapping and non-overlapping pairs, use : 

```bash
python preprocessing/gen_all_pairs_fileset.py
```
This will create a fileset with the same number of randomly chosen non-overlapping pairs from the generated subscans as overlapping pairs generated before during subscan generation.

Usage on **Predicted Scene Graphs** : Coming Soon! 

### Training :bullettrain_side:
To train SGAligner on 3RScan subscans generated from [here](#data--benchmark-hammer), you can use :

```bash
cd src
python trainers/trainval_sgaligner.py
```
We provide config files for the corresponding data in ``config/`` directory. Please change the parameters in the configuration files, if you want to tune the hyper-parameters.

### Evaluation :vertical_traffic_light:
#### Graph Alignment + Point Cloud Registration

```bash
cd src
python inference/sgaligner/inference_align_reg.py --config ../configs/scan3r/scan3r_ground_truth.yaml --snapshot <path to SGAligner trained model> --reg_snapshot <path to GeoTransformer model trained on 3DMatch>
```

#### Finding Overlapping vs Non-Overlapping Pairs
:heavy_exclamation_mark: Run [Generating Overlapping and Non-Overlapping Subscan Pairs](#Generating-Overlapping-and-Non-Overlapping-Subscan-Pairs) before.

To run the inference, you need to:

```bash
cd src
python inference/sgaligner/inference_find_overlapper.py --config ../configs/scan3r/scan3r_gt_w_wo_overlap.yaml --snapshot <path to SGAligner trained model> --reg_snapshot <path to GeoTransformer model trained on 3DMatch>
```

#### 3D Point Cloud Mosaicking
First, we generate the subscans per 3RScan scan using : 

```bash
python data-preprocessing/gen_scan_subscan_mapping.py --split <the split you want to generate the mapping for>
```

And then, to run the inference, you need to:

```bash
cd src
python inference/sgaligner/inference_mosaicking.py --config ../configs/scan3r/scan3r_gt_mosaicking.yaml --snapshot <path to SGAligner trained model> --reg_snapshot <path to GeoTransformer model trained on 3DMatch>
```

## TODO :soon:
- [X] ~~Add 3D Point Cloud Mosaicking~~
- [X] ~~Add Support For [EVA](https://github.com/cambridgeltl/eva)~~
- [ ] Add visualisation for registration results
- [ ] Add usage on Predicted Scene Graphs
- [ ] Add scene graph alignment of local 3D scenes to prior 3D maps
- [ ] Add overlapping scene finder with a traditional retrieval method (FPFH + VLAD + KNN)


## BibTeX :pray:
```
@article{,
  title     = {{}},
  author    = {},
  booktitle = {{}},
  year      = {}
}
```
## Acknowledgments :recycle:
In this project we use (parts of) the official implementations of the following works and thank the respective authors for open sourcing their methods: 

- [SceneGraphFusion](https://github.com/ShunChengWu/3DSSG) (3RScan Dataloader)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) (Registration)
- [MCLEA](https://github.com/lzxlin/MCLEA) (Alignment)
