
# DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions
This paper was submitted to ICASSP 2025.

## Status
This project is currently under active development. We are continuously updating and improving it, with more usage details and features to be released in the future.

# Getting started

## Download dataset and checkpoints 
1. Download the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset and place the dataset into `data/dataset` with structure looks like below:
```plaintext
data/dataset/LJSpeech-1.1
 ┣ metadata.csv
 ┣ wavs
 ┃ ┣ LJ001-0001.wav
 ┃ ┣ LJ001-0002.wav 
 ┃ ┣ ...
 ┣ README
```
2. Download the alignments of the LJSpeech dataset [LJSpeech.zip](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4). You have to unzip the files in `data/dataset/LJSpeech-1.1`
3. Download checkpoints (Coming Soon)
4. Uncompress the checkpoint tar file and place the content into **data/checkpoints/**

## Preprocessing
```python
python preprocessing.py
```

## Training

Train the VAE (Optional)
```python
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/autoencoder.py -c drawspeech/config/vae_ljspeech_22k.yaml
```

If you don't want to train the VAE, you can just use the VAE checkpoint that we provide.
- set the variable `reload_from_ckpt` in `drawspeech_ljspeech_22k.yaml` to `data/checkpoints/vae.ckpt`

Train the DrawSpeech
```python
CUDA_VISIBLE_DEVICES=0 python drawspeech/train/latent_diffusion.py -c drawspeech/config/drawspeech_ljspeech_22k.yaml
```


## Inference

If you have trained the model using `drawspeech_ljspeech_22k.yaml`, use the following syntax:
```shell
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml --list_inference tests/inference.json
```

If not, please specify the DrawSpeech checkpoint:
```shell
CUDA_VISIBLE_DEVICES=0 python drawspeech/infer.py --config_yaml drawspeech/config/drawspeech_ljspeech_22k.yaml --list_inference tests/inference.json --reload_from_ckpt data/checkpoints/drawspeech.ckpt
```

## Acknowledgement
This repository borrows codes from the following repos. Many thanks to the authors for their great work.
AudioLDM: https://github.com/haoheliu/AudioLDM-training-finetuning?tab=readme-ov-file#prepare-python-running-environment 
FastSpeech 2: https://github.com/ming024/FastSpeech2 
HiFi-GAN: https://github.com/jik876/hifi-gan 

