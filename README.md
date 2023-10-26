# Modeling Dynamics over Meshes with Gauge Equivariant Message Passing

Official implementation of `Modeling Dynamics over Meshes with Gauge Equivariant Message Passing`.

## Acknowledgements

The repo was built on top of the official [EMAN](https://github.com/gallego-posada/eman) and [GemCNN](https://github.com/Qualcomm-AI-research/gauge-equivariant-mesh-cnn) repos but was heavily modified. Please see `LICENSE_EMAN` for code relating to EMAN and `LICENSE_GemCNN` for code relating to GemCNN.

## Requirements

Install the system dependencies first (`cmake` for openmesh, `gfortran` for escnn). The commands listed below are for Ubuntu 22.04.

```bash
sudo apt install cmake gfortran 
```

To install the python dependencies, run the following commands. You may need to install `numpy` and `torch` before installing `torch-geometric`.

```bash
pip install -r requirements.txt
```

This was code was tested on the following specs:
* Ubuntu 22.04
* Python 3.10
* PyTorch 1.13.1
* PyTorch Geometric 2.3.1

## Usage

This repo uses [Hydra](https://hydra.cc/) for config management (yaml files are inside `conf/`).  See `conf/backbone` to see possible backbones (GemCNN, EMAN, Hermes), `conf/dataset` to see possible datasets, and `conf/dataset_backbone` to see dataset-specific backbone parameters.

We use [WandB](https://wandb.ai/) to track experiments and [PyTorch-Ignite](https://pytorch.org/ignite/index.html) to run experiments.

Before using, please install the dependencies and install this repo

```bash
pip install -e .
```

## Datasets

The datasets folder is `data/` and the raw data should be placed in `data/<Dataset name>/raw` for each dataset.

<details><summary>Details</summary>

### PDE
To generate the heat data, run `python src/data/pde/generate_heat.py`. You can optionally set `plot = True` to see visualizations.
To generate the wave data, run `python src/data/pde/generate_wave.py`. You can optionally set `plot = True` to see visualizations.
To generate the fineness and roughness datasets, run `python src/data/pde/generate_single_heat.py` and `python src/data/pde/generate_single_wave.py`.

### FAUST
You can download the FAUST dataset at <https://faust-leaderboard.is.tuebingen.mpg.de/>. Place `MPI-FAUST.zip` inside `data/FAUST/raw`.

### Objects
To generate the objects interacting on a mesh train and test datasets, run `bash scripts/generate_mesh_objects.sh`.  The datasets will be created under `data/objects/N250_O50/raw`.


</details>

## Training

This repo uses `hydra` and `wandb` to track experiments.
See `conf/backbone` to see possible backbones (GemCNN, EMAN, Hermes), `conf/dataset` to see possible datasets, and `conf/dataset_backbone` to see dataset-specific backbone parameters.

Run the following command:
```
python experiments/train.py dataset=heat backbone=hermes
```

<details><summary>Details</summary>

`dataset` values:
- faust
- objects
- heat
- wave
- cahn_hilliard

`backbone` values:
- gem_cnn
- eman
- hermes

Additionally, you can run the mesh fineness and roughness experiments by setting `dataset=heat_other` or `dataset=wave_other` and changing `dataset.cls.root` to the correct data folder path (e.g. `data/fineness/heat/reduce_0.99/raw`).

</details>

## Pretrained Model Checkpoints

Pretrained model checkpoints are provided in `pretrained_checkpoints` for all datasets for GemCNN, EMAN, and Hermes.

## Evaluation on rollouts

Using the pretrained model checkpoints, you can generate prediction rollouts for the PDE datasets. Run the command:

```
python experiments/eval_rollout.py dataset=heat backbone=hermes model_save_path=pretrained_checkpoints/Heat_Hermes_model.pt
``` 
using `model_save_path` as the path to the saved checkpoint file. 

<details><summary>Details</summary>

The `eval_rollout.py` creates outputs inside the `rollouts/` directory. It creates 3 files `losses.npy`, `predictions.npy`, and `ground_truth.npy` which are `numpy` arrays of losses, model predictions, and ground truth values. It additionally creates some visualizations of the ground truths and the model predictions.

`dataset` values:
- heat
- wave
- cahn_hilliard

`backbone` values:
- gem_cnn
- eman
- hermes


Ensure that the value for `dataset` and `backbone` match the model checkpoint. See the file `conf/eval_rollout.yaml` for configuration.

</details>

