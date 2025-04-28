# GRIT Reimplementation based on the Machine Learning Reproducibility Challenge (MLRC)

[![arXiv](https://img.shields.io/badge/arXiv-2305.17589-b31b1b.svg)](https://arxiv.org/abs/2305.17589)
[![arXiv](https://img.shields.io/badge/arXiv-2305.17589-b31b1b.svg)](https://arxiv.org/abs/2305.17589)

This repository contains a partial reimplementation of the model proposed in the paper:

> **Graph Inductive Biases in Transformers without Message Passing**  
> Ma et al., 2024 ([arXiv link](https://arxiv.org/abs/2305.17589))

This work was completed as part of a course project, inspired by the Machine Learning Reproducibility Challenge (MLRC) initiative.  
Our goal was to reproduce key aspects of the GRIT model based **solely** on the paper description, without referring to the official codebase.

## Implemented components

The reimplemented parts include:
- **Relative Random Walk Probabilities (RRWP)** computation in [`graphgps/transform/rrwp.py`](graphgps/transform/rrwp.py)
- **Flexible attention mechanism** as described in the paper, implemented in [`graphgps/layer/grit_layer.py`](graphgps/layer/grit_layer.py) under `MultiHeadAttentionLayerGrit`
- **Transformer layers with degree injection** implemented in [`graphgps/layer/grit_layer.py`](graphgps/layer/grit_layer.py) under `GritTransformerLayer`

**Note:**  
- The **node and edge encoders** using RRWP (`graphgps/encoder/grit_encoder.py`) and the **global Transformer model structure** (`graphgps/network/grit_model.py`) were kept almost identical to the original GraphGPS code, as the paper provided very limited information about these components or they remained largely standard.

## Environment setup

We recommend setting up the environment using Conda:

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

# For handling molecular datasets
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```

## Running the experiment on ZINC

```bash
conda activate graphgps

python main.py --cfg configs/GRIT/zinc-GRIT.yaml wandb.use False
```

This command trains the reimplemented GRIT model on the ZINC dataset with the standard configurations given in the paper. 

## Project Base and References

This project is a fork of the [GraphGPS repository](https://github.com/rampasek/GraphGPS), which provides a general, powerful, and scalable framework for graph Transformers. We used GraphGPS codebase as the foundation for our reimplementation of the GRIT model.

For more information please refer to:

- **GraphGPS GitHub Repository**: [https://github.com/rampasek/GraphGPS](https://github.com/rampasek/GraphGPS)
- **GRIT GitHub Repository**: [https://github.com/GRIT/GRIT](https://github.com/GRIT/GRIT)
