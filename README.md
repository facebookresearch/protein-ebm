# Energy-based models for atomic-resolution protein conformations
Pytorch implementation of [Energy-based models for atomic-resolution protein conformations](https://openreview.net/forum?id=S1e_9xrFvS), accepted to ICLR 2020 (with spotlight). Includes training code, models, datasets, and pre-trained model weights.

![Framework overview](https://dl.fbaipublicfiles.com/protein-ebm/framework_overview.png)

## Dependencies

To install the dependencies for the project, execute
```
pip install -r requirements.txt
```

For reproducibility, we listed the packages used for generating the results in the paper, but other versions of these packages will likely give similar results.

## Downloading the datasets

### 1. Rotamer Library
First, get the [Dunbrack rotamer library](http://dunbrack.fccc.edu/bbdep2010/) (935.0 MB), which is used for negative sampling.
```
wget https://dl.fbaipublicfiles.com/protein-ebm/dunbrak_rotamer.tar.gz
tar xvzf dunbrack_rotamer.tar.gz 
```
If necessary, update `ROTAMER_LIBRARY_PATH` in `config.py` to point to the `original/` directory in the uncompressed `tar.gz` parent directory.

### 2. Protein structures
Next, download the mmCIF files (38.7 GiB) used for training:
```
wget https://dl.fbaipublicfiles.com/protein-ebm/mmcif.tar.gz
tar xvzf mmcif.tar.gz
```
If necessary, update `MMCIF_PATH` in `config.py` to point to the unzipped directory.

### 3. [optional] Pre-trained models
To retreive pre-trained EBM models (48.9 MB) that were evaluated in [our paper](https://openreview.net/forum?id=S1e_9xrFvS), run
```
wget https://dl.fbaipublicfiles.com/protein-ebm/cachedir.tar.gz
tar xvzf cachedir.tar.gz
```

The transformer model evaluated in the paper can then be found at `cachedir/transformer_gmm_uniform/model_130000`. 

## Preprocessing
After downloading the protein structures, please preprocess the data by running
```
python mmcif_utils.py
```
This will generate a set of preprocessed intermediates for each mmCIF file. Additionally, it run a set of unit tests to check correctness of the kinematics operations.

## Training

### Negative Sampling Arguments

EBMs are trained with positive samples and negative samples. The positive samples are the mmCIF files you generated above, but the negative samples are chosen or constructed during training. Our codebase implements several different strategies for negative sampling:
1. uniform: when selected, sample the rotamers uniformly from the Dunbrack rotamer library. If not, sample the rotamers based on their emperical frequencies.
2. weighted\_gauss: when selected, interpolate between candidate rotamers for nearby phi/psi bins by sampling from a Gaussian using weighted means and variances of each nearby bin
3. gmm: when selected, interpolate between candidate rotamers for nearby phi/psi bins by sampling from a Gaussian Mixture Model with mixture weights corresponding to the distances of each bin
4. chi\_mean: when selected, sample candidate rotamers by using mean chi angles from the Dunbrack library and angles 1 standard deviation away

For the experiments reported in [our paper](https://openreview.net/forum?id=S1e_9xrFvS), we interpolated between candidate rotamers using a Guassian Mixture Model, with uniform sampling of the rotamers. This corresponds to the negative sample arguments: `-gmm --uniform`.

### Model Arguments

In [our paper](https://openreview.net/forum?id=S1e_9xrFvS), we report the performance of various model architectures for predicting energies of rotamer configurations. The models we tested include:
1. fc: a MLP architecture
2. s2s: a Set2Set architecture
3. transformer: a transformer architecture
2. graph: a GNN architecture

For our experiments and the released pre-trained model, we use the transformer architecture, which corresponds to `--model transformer`. 

### Training the model
The following script will train a model on a single node:
```
python train.py

## Negative sampling arguments
--neg-sample 1                     # number of negative rotamer samples for rotamer trials (1-1 ratio)
--gmm                              # example negative sampling arugment
--uniform                          # example negative sampling argument

## Model parameters
--encoder-layers 3                 # number of layers

## optimization
--gpus 1                           # number of gpus
--nodes 1                          # number of nodes
--batch-size 196                   # batch size
--end-lr 0.0002                    # learning rate after warmup
--nodes 1

## main parameters
--exp test_ebm                     # experiment name
```

In [our paper](https://openreview.net/forum?id=S1e_9xrFvS), we trained models on 32 different GPUs. Therefore, we've included code in this release for distributed training on multiple nodes. The following script will train a model on 32 GPUs:
```
python train.py

## Negative sampling arguments
--neg-sample 1                     # number of negative rotamer samples for rotamer trials (1-1 ratio)
--gmm                              # example negative sampling arugment
--uniform                          # example negative sampling argument

## Model parameters
--encoder-layers 3                 # number of layers

## optimization
--gpus 1                           # number of gpus
--nodes 4                          # number of nodes
--batch-size 196                   # batch size
--end-lr 0.0002                    # learning rate after warmup
--nodes 1

## distributed training
--node-rank 2                      # the rank of the current node
--master-addr 8.8.8.8              # sets os.environ['MASTER_ADDR']
--master-port 8888                 # sets os.environ['MASTER_PORT']
--slurm                            # if using SLURM

## main parameters
--exp test_ebm                     # experiment name
```

To enable distributed training, set `master-addr` and `master-port` to an accessible machine. Then, run one copy of this script on each node, ensuring that the process has access to all GPUs on that node. Gradients will sync to the `master-addr` and `master-port`.

If you are using SLURM, include the `--slurm` argument. This tells `torch.distributed` to set `init_method='env://'`. Otherwise, `init_method=tcp://localhost:1492` by default.

If you need to modify any of the distributed settings for your infrastructure, see the start of the `main()` and `main_single()` functions in `train.py` where this logic is implemented.


## Visualization
To reproduce the visualizations shown in our paper, run
```
python vis_sandbox.py --exp=transformer_gmm_uniform --resume-iter=130000 --task=[new_model, tsne]
```

where the task can be switched to visualize different results. 

Additional code for creating saliency maps can be found in `scripts/extract_saliency.py` and `scripts/generate_colormap_saliency.py`.


## Quantitative Metrics
To produce the quantiative metrics for performance on the rotamer recovery task (using discrete sampling),
run
```
python vis_sandbox.py --exp=transformer_gmm_uniform --resume-iter=130000 --task=rotamer_trial --sample-mode=rosetta --neg-sample 500 --rotations 10
```
or run
```
python vis_sandbox.py --exp=transformer_gmm_uniform --resume-iter=130000 --task=rotamer_trial --sample-mode=gmm --neg-sample 150  --rotations 10
```
for continuous sampling.

## Citing this work
If you find the EBM useful, please cite [our corresponding paper](https://openreview.net/forum?id=S1e_9xrFvS). Additionally, significant time and effort went into the dataset construction established by previous works. Please cite those papers if you use the associated datasets.

### Our paper (Protein EBM):
```
@inproceedings{
    Du2020Energy-based,
    title={Energy-based models for atomic-resolution protein conformations},
    author={Yilun Du and Joshua Meier and Jerry Ma and Rob Fergus and Alexander Rives},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=S1e_9xrFvS}
}
```

### Dunbrack rotamer library:
```
@article{bower1997prediction,
  title={Prediction of protein side-chain rotamers from a backbone-dependent rotamer library: a new homology modeling tool},
  author={Bower, Michael J and Cohen, Fred E and Dunbrack Jr, Roland L},
  journal={Journal of molecular biology},
  volume={267},
  number={5},
  pages={1268--1282},
  year={1997},
  publisher={Elsevier}
}
```

### CullPDB Database
```
@Article{dunbrackresolution,
   Author="G. Wang and R. L. Dunbrack, Jr.",
   Title="PISCES: a protein sequence culling server",
   Journal="Bioinformatics",
   Year="2003",
   Volume="19",
   Pages="1589-1591"
}
```

### Rotamer recovery benchmark
```
@incollection{leaver2013scientific,
  title={Scientific benchmarks for guiding macromolecular energy function improvement},
  author={Leaver-Fay, Andrew and O'Meara, Matthew J and Tyka, Mike and Jacak, Ron and Song, Yifan and Kellogg, Elizabeth H and Thompson, James and Davis, Ian W and Pache, Roland A and Lyskov, Sergey and others},
  booktitle={Methods in enzymology},
  volume={523},
  pages={109--143},
  year={2013},
  publisher={Elsevier}
}
```

## License
protein-ebm is MIT licensed, as found in the LICENSE file in the root directory of this source tree.
