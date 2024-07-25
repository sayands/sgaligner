<div align='center'>
<h2 align="center"> SGAligner : 3D Scene Alignment with Scene Graphs </h2>
<h3 align="center">ICCV 2023</h3>

<a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup>1</sup>, <a href="https://miksik.co.uk/">Ondrej Miksik</a><sup>2</sup>, <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>, <a href="https://www.linkedin.com/in/d%C3%A1niel-bar%C3%A1th-3a489092/">Daniel Barath</a><sup>1</sup>, <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft Mixed Reality & AI Labs

SGAligner aligns 3D scene graphs of environments using multi-modal learning and leverage the output for the downstream task of 3D point cloud registration.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sgaligner-3d-scene-alignment-with-scene/3d-scene-graph-alignment-on-3dssg)](https://paperswithcode.com/sota/3d-scene-graph-alignment-on-3dssg?p=sgaligner-3d-scene-alignment-with-scene)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/sgaligner-3d-scene-alignment-with-scene/point-cloud-registration-on-3rscan)](https://paperswithcode.com/sota/point-cloud-registration-on-3rscan?p=sgaligner-3d-scene-alignment-with-scene)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

![teaser](https://sayands.github.io/sgaligner/static/images/teaser.png)


</div>

[[Project Webpage](https://sayands.github.io/sgaligner/)]
[[Paper](https://arxiv.org/abs/2304.14880)]


## News :newspaper:

* **14. July 2023** : SGAligner accepted to ICCV 2023. :fire:
* **1. May 2023**: [SGAligner preprint](https://arxiv.org/abs/2304.14880v1) released on arXiv.
* **10. April 2023**: Code released.

## Code Structure :clapper:

```
├── sgaligner
│   ├── example-data                  <- examples of data generated post preprocessing
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
> Note: Please follow [PointCloudTransformer](https://github.com/qinglew/PointCloudTransformer) repository for installing pointnet2_ops_lib if you want to use PCT as the backbone.

Please follow the submodule for additional installation requirements and setup of [GeoTransformer](https://github.com/sayands/GeoTransformer).

### Downloads :droplet:
The pre-trained model and other meta files are available [here](https://drive.google.com/drive/folders/1elqdbYD5T2686r42lcUHE6SyiFnDsZur?usp=sharing).

### Dataset Generation :hammer:
After installing the dependencies, we preprocess the datasets and provide the benchmarks. 

#### Subscan Pair Generation - 3RScan + 3DSSG
Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Move all files of 3DSSG to a new ``files/`` directory within Scan3R. The structure should be:

```
├── 3RScan
│   ├── files       <- all 3RScan and 3DSSG meta files (NOT the scan data)  
│   ├── scenes      <- scans
│   └── out         <- Default output directory for generated subscans (created when running pre-processing)
```

> To generate ``labels.instances.align.annotated.v2.ply`` for each 3RScan scan, please refer to the repo from 
[here](``https://github.com/ShunChengWu/3DSSG/blob/master/data_processing/transform_ply.py``).

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
To train SGAligner on 3RScan subscans generated from [here](#dataset-generation-hammer), you can use :

```bash
cd src
python trainers/trainval_sgaligner.py --config ../configs/scan3r/scan3r_ground_truth.yaml
```

#### EVA Training
We also provide training scripts for [EVA](https://arxiv.org/abs/2009.13603), used as a baseline after being adapted for scene graph alignment. To train EVA similar to SGAligner on the same data, you can use :

```bash
cd src
python trainers/trainval_eva.py --config ../configs/scan3r/scan3r_eva.yaml
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

## Benchmark :chart_with_upwards_trend:
We provide detailed results and comparisons here.

### 3D Scene Graph Alignment (Node Matching)
| Method | Mean Reciprocal Rank | Hits@1 | Hits@2 | Hits@3 | Hits@4 | Hits@5 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [EVA](https://github.com/cambridgeltl/eva) | 0.867 | 0.790 | 0.884 | 0.938 | 0.963 | 0.977 | 
| $\mathcal{P}$ | 0.884 | 0.835 | 0.886 | 0.921 | 0.938 | 0.951 |
| $\mathcal{P}$ + $\mathcal{S}$ | 0.897 | 0.852 | 0.899 | 0.931 | 0.945 | 0.955 |
| $\mathcal{P}$ + $\mathcal{S}$ + $\mathcal{R}$ | 0.911 | 0.861 | 0.916 | 0.947 | 0.961 | 0.970 |
| SGAligner | 0.950 | 0.923 | 0.957 | 0.974 | 0.9823 | 0.987 |

### 3D Point Cloud Registration
| Method | CD | RRE | RTE | FMR | RR |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [GeoTr](https://github.com/qinzheng93/GeoTransformer) | 0.02247	| 1.813 | 2.79 | 98.94 | 98.49 |
| Ours, K=1 | 0.01677 | 1.425 | 2.88 | 99.85 | 98.79 |
| Ours, K=2 | 0.01111 | 1.012 | 1.67 | 99.85 | 99.40 |
| Ours, K=3 | 0.01525 | 1.736 | 2.55 | 99.85 | 98.81 | 

## TODO :soon:
- [X] ~~Add 3D Point Cloud Mosaicking~~
- [X] ~~Add Support For [EVA](https://github.com/cambridgeltl/eva)~~
- [ ] Add usage on Predicted Scene Graphs
- [ ] Add scene graph alignment of local 3D scenes to prior 3D maps
- [ ] Add overlapping scene finder with a traditional retrieval method (FPFH + VLAD + KNN)


## BibTeX :pray:
```bibtex
@inproceedings{Sarkar_2023_ICCV,
    author    = {Sarkar, Sayan Deb and Miksik, Ondrej and Pollefeys, Marc and Barath, Daniel and Armeni, Iro},
    title     = {SGAligner: 3D Scene Alignment with Scene Graphs},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {21927-21937}
}
```
## Acknowledgments :recycle:
In this project we use (parts of) the official implementations of the following works and thank the respective authors for open sourcing their methods: 

- [SceneGraphFusion](https://github.com/ShunChengWu/3DSSG) (3RScan Dataloader)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) (Registration)
- [MCLEA](https://github.com/lzxlin/MCLEA) (Alignment)
