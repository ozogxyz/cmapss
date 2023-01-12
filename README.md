______________________________________________________________________

# Hybrid Deep Learning Algorithms for Multivariate Time Series Forecasting (NASA C-MAPSS Benchmark)

<div align="center">

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)

</div>

______________________________________________________________________

## Description

Novel algorithms to predict Remaining Useful Life (RUL) on NASA’s benchmark dataset, CMAPSS turbofan engine degradation simulation.

## Dataset

<div align="justify">

Benchmark source: NASA Intelligent Systems Division: Prognostics Center of Excellence - Prognostic Health Management, Predictive Maintenance of Turbofan Engines.

The generation of data-driven prognostics models requires the availability of data sets with run-to-failure trajectories. To contribute to the development of these methods, the data set provides a new realistic data set of run-to-failure trajectories for a small fleet of aircraft engines under realistic flight conditions. The damage propagation modelling used for the generation of this synthetic data set builds on the modeling strategy from previous work. The data set was generated with the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dynamical model. The data set has been provided by the NASA Prognostics Center of Excellence (PCoE) in collaboration with ETH Zurich and PARC.

</div>

[Link to the dataset](https://www.nasa.gov/intelligent-systems-division/)

Download Mirror: https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

<div align="justify">
Data Set Citation: M. Chao, C.Kulkarni, K. Goebel and O. Fink (2021). "Aircraft Engine Run-to-Failure Dataset under real flight conditions",
NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
</div>
______________________________________________________________________

## Project Description

<div align="justify">

This project uses PyTorch Lightning powered rul-datasets to generate data sets. Hydra is an open source powerful utility to configure
experiments and generate configuration files. The template has many loggers to choose from including popular MLFlow and Weights and Biases.

</div>
______________________________________________________________________

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/ozogxyz/cmapss
cd cmapss

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
