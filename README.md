## SGAligner : 3D Scene Alignment with Scene Graphs
<div align='center'>

<a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup>1</sup>, <a href="https://miksik.co.uk/">Ondrej Miksik</a><sup>2</sup>, <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>, <a href="https://www.linkedin.com/in/d%C3%A1niel-bar%C3%A1th-3a489092/">Daniel Barath</a><sup>1</sup>, <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft Mixed Reality & AI Labs

</div>

![teaser](https://sayands.github.io/sgaligner/static/images/teaser.png)
SGAligner aligns 3D scene graphs of environments using multi-modal learning and leverage the output for the downstream task of 3D point cloud registration.

## News :newspaper:

* **10. April 2023**: Code + [SGAligner Preprint]() released on arXiv.

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
│   │   │── inference                 <- inference files for the downstream applications
│   │   │── trainval.py               <- train loop
│   │── utils                         <- utils
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
git clone git@github.com:sayands/sgaligner.git
cd sgaligner
conda env create -f req.yml
```

Please follow the submodule for additional installation requirements of [GeoTransformer](https://github.com/sayands/GeoTransformer).

### Data + Benchmark :hammer:
After installing the dependencies, we preprocess the datasets and provide the benchmarks. 

#### Subscan Pair Generation - 3RScan + 3DSSG
Download [3RScan](https://github.com/WaldJohannaU/3RScan) and [3DSSG](https://3dssg.github.io/). Move all files of 3DSSG to a new directory within Scan3R. The structure should be:

```
├── 3RScan
│   ├── files       <- all 3RScan and 3DSSG meta files (NOT the scan data)  
│   ├── scenes      <- scans
│   ├── out         <- Default output directory for generated subscans (created when running pre-processing)
```

Change the absolute paths in ``utils/define.py``.

First, we create sub-scans from each 3RScan scan using the ground truth scene Graphs from the 3DSSG dataset and then calculate the pairwise overlap ratio for the subscans in a scan. Finally, we preprocess the data for our framework. The relevant code can be found in the ``data-preprocessing/`` directory. You can use the following command to generate the subscans.

```bash
bash scripts/generate_data_scan3r_gt.sh
```
To adhere to our evaluation procedure, please do not change the seed value in the files in ``configs/`` directory. 

#### Finding Overlapping vs Non-Overlapping Subscenes
To generate non-overlapping pairs, use : 

```bash
python data-preprocessing/gen_all_pairs_fileset.py --config 
```
This will generate exactly the same number of non-overlapping pairs as overlapping pairs generated before during subscan generation.


Usage on Predicted Scene Graphs : Coming Soon! 

### Training :bullettrain_side:
To train SGAligner, you can use :

```bash
cd src
python trainval.py --config_file <config_file_name>
```
We provide config files for the corresponding data in ``config/`` directory. Please change the parameters in the configuration files, if you want to tune the hyper-parameters.

### Evaluation :vertical_traffic_light:
#### Graph Alignment + Point Cloud Registration

```bash
cd src
python inference/inference_align_reg.py --snapshot <path to SGAligner trained model> --reg_snapshot <path to GeoTransformer model trained on 3DMatch>
```

#### Finding Overlapping vs Non-Overlapping Scans
Change ``_C.data.anchor_type_name`` in the corresponding configuration file to ``_subscan_anchors_w_wo_overlap`` for running this inference. To run the inference, you need to:

```bash
cd src
python inference/inference_find_overlapper.py --snapshot <path to SGAligner trained model> --reg_snapshot <path to GeoTransformer model trained on 3DMatch>
```

## BibTeX :pray:
```
@article{,
  title     = {{}},
  author    = {},
  booktitle = {{}},
  year      = {}
}
```

### TODO
- [ ] Add usage on Predicted Scene Graphs
- [ ] Provide a script to port the predicted scene graphs to the ground truth 3RScan format.


[//]: <> (We also show results on sub-scans generated using predicted scene graphs, please refer to the 3DSSG repository for the graph prediction process.)

