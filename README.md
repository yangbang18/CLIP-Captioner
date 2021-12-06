# CLIP-Captioner

PyTorch Implementation of the paper

> **CLIP Meets Video Captioners: Attribute-Aware Representation Learning Promotes Accurate Captioning**
>
> Bang Yang and Yuexian Zou\*.
>
> [[ArXiv](https://arxiv.org/abs/2111.15162)]


## TOC

  - [Environment](#environment)
  - [Corpora/Feature Preparation](#corporafeature-preparation)
  - [Training](#training)
  - [Testing](#testing)
  - [Analysis/Examples](#analysisexamples)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)


## Environment

1. Clone the repo:

```
git clone git@github.com:yangbang18/CLIP-Captioner.git --recurse-submodules
```

2. Use Anaconda to create a new environment:

```
conda create -n vc python==3.7
conda activate vc
```

3. Install necessary packages:

```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.5.1
pip install pandas h5py nltk pillow wget
```

Here we use `torch 1.7.1` based on `CUDA 10.1`. Lower versions of `torch` are imcompatible with the submodule `CLIP`.

## Corpora/Feature Preparation

* Following the structure below to place corpora and feature files:
  ```
  └── base_data_path
      ├── MSRVTT
      │   ├── feats
      │   │   ├── image_R101_fixed60.hdf5
      │   │   ├── CLIP_RN50.hdf5
      │   │   ├── CLIP_RN101.hdf5
      │   │   ├── CLIP_RN50x4.hdf5
      │   │   ├── CLIP_ViT-B-32.hdf5
      │   │   ├── motion_resnext101_kinetics_fixed60.hdf5
      │   │   └── audio_vggish_audioset_fixed60.hdf5
      │   ├── info_corpus.pkl
      │   └── refs.pkl
      └── VATEX
          ├── feats
          │   ├── ...
          │   └── audio_vggish_audioset_fixed60.hdf5
          ├── info_corpus.pkl
          └── refs.pkl
  ```

**Please remember to modify `base_data_path` in [config/Constants.py](config/Constants.py)**

Preparing data on your own:

1. Preprocessing corpora:
   ```
   python pretreatment/prepare_corpora.py --dataset MSRVTT --sort_vocab --attribute_first
   python pretreatment/prepare_corpora.py --dataset VATEX --sort_vocab --attribute_first
   ```
2. Feature extraction:

* Downloading all video files of [MSRVTT](http://ms-multimedia-challenge.com/2017/dataset) and VATEX
* Extracting frames

  ```
  python pretreatment/extract_frames_from_videos.py \
  --video_path $path_to_video_files \
  --frame_path $path_to_save_frames \
  --video_suffix mp4 \
  --frame_suffix jpg \
  --strategy 0
  ```
* Extracting image features of INP models

  ```
  python pretreatment/extract_image_feats_from_frames.py \
  --frame_path $path_to_load_frames \
  --feat_path $base_data_path/$dataset_name \
  --feat_name image_R101_fixed60.hdf5 \
  --model resnet101 \
  --frame_suffix jpg
  --gpu 0
  ```
* Extracting image features of CLIP models

  ```
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN50
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN101
  python pretreatment/clip_feats.py --dataset MSRVTT --arch RN50x4
  python pretreatment/clip_feats.py --dataset MSRVTT --arch ViT-B/32
  ```
* Extracting motion and audio features: comming soon

## Training

```
python train.py \
--dataset $dataset_name \
--method $method_name \
--feats $feats_name \
--modality $modality_combination \
--arch $arch_name \
--task $task_name
```

1. supported datasets

- `MSRVTT`
- `VATEX`

2. supported methods, whose configurations can be found in [config/methods.yaml](config/methods.yaml)

- `Transformer`: our baseline (autoregressive decoding)
- `TopDown`: a two layer LSTM decoder (autoregressive decoding)
- `ARB`: a slight different encoder (autoregressive decoding)
- `NACF`: a slight different encdoer (non-autoregressive decoding)

3. supported feats, whose configurations can be found in [config/feats.yaml](config/feats.yaml)

- `R101`: ResNet-101 (INP)
- `RN50`: ResNet-50 (CLIP)
- `RN101`: ResNet-101 (CLIP)
- `RN50x4`: ResNet-50x4 (CLIP)
- `ViT`: ViT-B/32 (CLIP)
- `I3D`: used in VATEX

4. supported modality combinations: any combination of `a` (audio), `m` (motion) and `i` (image).
5. supported archs, whose configurations can be found in [config/archs.yaml](config/archs.yaml)

- `base`: used in MSRVTT
- `large`: used in VATEX

6. supported tasks, whose configurations can be found in [config/tasks.yaml](config/tasks.yaml)

- `diff_feats`: to test different combinations of modalities
- `VAP`: video-based attribute prediction
- `VAP_SS0`: `VAP` w/o sparse sampling
- `TAP`: text-based attribute prediction (used in Transformer-based methods)
- `TAP_RNN`: text-based attribute prediction (used in RNN-based methods)
- `DAP`: dual attribute prediction (used in Transformer-based methods)
- `DAP_RNN`: dual attribute prediction (used in RNN-based methods)

**Examples:**

```
python train.py --dataset MSRVTT --method Transformer --feats RN101 --modality ami --arch base --task diff_feats
python train.py --dataset MSRVTT --method Transformer --feats RN101 --modality mi --arch base --task diff_feats
python train.py --dataset MSRVTT --method Transformer --feats RN101 --modality i --arch base --task diff_feats

python train.py --dataset MSRVTT --method Transformer --feats ViT --modality ami --arch base --task diff_feats
python train.py --dataset MSRVTT --method Transformer --feats ViT --modality ami --arch base --task DAP

python train.py --dataset MSRVTT --method TopDown --feats ViT --modality ami --arch base --task diff_feats
python train.py --dataset MSRVTT --method TopDown --feats ViT --modality ami --arch base --task DAP_RNN
```

## Testing

```
python translate.py --checkpoint_paths $path_to_the_checkpoint
python translate.py --checkpoint_paths $path_to_the_checkpoint1 $path_to_the_checkpoint2  # ensembling
```

## Analysis/Examples

Please see the [notebooks](notebooks) folder.

## Citation

Please **[★star]** this repo and **[cite]** the following paper if you feel our code or models useful to your research:

```
@article{yang2021clip,
  title={CLIP Meets Video Captioners: Attribute-Aware Representation Learning Promotes Accurate Captioning},
  author={Yang, Bang and Zou, Yuexian},
  journal={arXiv preprint arXiv:2111.15162},
  year={2021}
}

@inproceedings{yang2021NACF,
  title={Non-Autoregressive Coarse-to-Fine Video Captioning}, 
  author={Yang, Bang and Zou, Yuexian and Liu, Fenglin and Zhang, Can},   
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={3119-3127},
  year={2021}
}
```

## Acknowledgements

- This codebase is built upon our previous one, i.e., [Non-Autoregressive-Video-Captioning](https://github.com/yangbang18/Non-Autoregressive-Video-Captioning).
- Code of the non-autoregressive decoding is based on [facebookresearch/Mask-Predict](https://github.com/facebookresearch/Mask-Predict).
