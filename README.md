# PS4 Dataset 🧬

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ps4-a-next-generation-dataset-for-protein/protein-secondary-structure-prediction-on-1)](https://paperswithcode.com/sota/protein-secondary-structure-prediction-on-1?p=ps4-a-next-generation-dataset-for-protein)

PS4 is the largest open-source dataset for Protein Single Sequence Secondary Structure prediction. 

This repository contains the data itself, methods to validate and add new samples to the dataset, and a toolkit for developing and evaluating state-of-the-art secondary structure prediction models.

For more information, see the official [paper](https://www.future-science.com/doi/10.2144/btn-2023-0024).

## 🤗 Hugging Face Space

If you're primarily interested in using the pretrained models to predict secondary structure on your own sequences, you can use this [official Hugging Face Space](https://huggingface.co/spaces/omarperacha/protein-secondary-structure-prediction) to do so with no setup.

## Data

The core dataset is contained in `ps4_data/data/data.csv`. This contains 18,731 proteins with their PDB code, index of the first residue in their respective DSSP file, their residue sequence and 9-category secondary structure sequence (including polyproline helices).

The train/test split can be found in `ps4_data/data/chain_ids.npz`.

## Installation

After cloning the repository, `cd` into the root directory and run:
```
chmod a+rx install.sh
./install.sh
```
This will make the install script executable and then proceed to run it, which will:
- Download all required git submodules
- Install all PyPI module dependencies
- Build and install [Mega](https://github.com/facebookresearch/mega)
- Build and install the `ps4-rs` Rust package, for preprocessing samples when making new additions to PS4, using [maturin](https://github.com/PyO3/maturin).

Building `ps4-rs` requires Rust to be installed. Please see the [official Rust language documentation](https://www.rust-lang.org/tools/install) for installation instructions.

## Usage

### Data Preparation

Pretrained weights are made available via git LFS. To evaluate with trained weights or tain a new model, first generate the pytorch-ready dataset:
```
python main.py --gen_dataset
```

Generating the pytorch-ready dataset involves extracting embeddings from a large pretrained protein language model and may take a few hours. It is **highly recommended to run this process on GPU.** 

### Training

Once `--gen_dataset` has been run, you can train a new model from scratch. To train PS4-Mega: 
```
python main.py --train --mega
```
To train PS4-Conv: 
```
python main.py --train --conv
```
To train a custom model, add your model code and make the relevant adjustments to `ps4_train/train.py`.

### Evaluation

Once you've run `--gen_dataset`, you can easily evaluate pretained models on the PS4 test set or the CB513. For example, to evaluate PS4-Mega on CB513:
```
python main.py --eval --cb513 --mega
```
Or to evaluate PS4-Conv on the PS4 test set:
```
python main.py --eval --ps4 --conv
```
You can optionally specify a path to your own model weights as a final argument to `python main.py --eval` to evaluate a custom model.

### Predicting Secondary Structure for New Sequences 

You can use our pretrained models to predict secondary structure on new sequences listed in a FASTA file:
```
python main.py --sample <fasta_path>
```
Or to specify to use PS4-Conv instead of PS4-Mega:
```
python main.py --sample <fasta_path> --conv
```
Sampling also involves extracting embeddings from a large pretrained protein language model. It is recommended to run this process on a machine with at least 16GB of RAM.  

### Extending the Dataset

With the help of the community, we can continue to grow PS4 and ensure a high-quality, non-redundant dataset. To preprocess new samples to add them to the PS4, run the following to ensure non-redundancy against the rest of the PS4 dataset and the CB513 using the `ps4-rs` package:
```
python extend_ps4.py <in_path> <out_path>
```

where `<in_path>` is replaced by the path to a folder containing DSSP files of proteins you'd like to add to the dataset, and `<out_path>` is the file you'd like to save the new non-redundant samples to, and must end in `.csv`.

To share new samples with the community, open a pull request!
