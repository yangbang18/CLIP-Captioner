# CLIP-Captioner

PyTorch Implementation of the paper

> **CLIP Meets Video Captioning: Concept-Aware Representation Learning Does Matter**
>
> Bang Yang, Tong Zhang and Yuexian Zou\*.
>
> [[Springer](https://link.springer.com/chapter/10.1007/978-3-031-18907-4_29)], [[ArXiv](https://arxiv.org/pdf/2111.15162)]


## TOC

- [CLIP-Captioner](#clip-captioner)
  - [TOC](#toc)
  - [Update Notes](#update-notes)
  - [Environment](#environment)
  - [Corpora/Feature Preparation](#corporafeature-preparation)
  - [Training](#training)
  - [Testing](#testing)
  - [Analysis/Examples](#analysisexamples)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Update Notes
[2023-03-16] Add guideline for audio feature extraction; Update links for downloading pre-extracted features

[2023-02-23] Release the "dust-laden" code

## Environment

1. Clone the repo:

```
git clone https://github.com/yangbang18/CLIP-Captioner.git --recurse-submodules

# Alternatively
git clone https://github.com/yangbang18/CLIP-Captioner.git
git submodule update --init
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

Due to the legal and privacy concerns, we cannot directly share the downloaded videos or clips from YouTube in any way. Instead, we share the pre-processed files and pre-extracted feature files: [MSRVTT](https://pkueducn-my.sharepoint.com/:f:/g/personal/2101112290_pku_edu_cn/EkST0Ik4tpFJhbWqb70zdq4BB-LgXxuIKvER5_lxGSIMaw?e=raAeNs), [VATEX](https://pkueducn-my.sharepoint.com/:f:/g/personal/2101112290_pku_edu_cn/Er_ttYtTzYNOgApvEWLyJTIBM_RTV9xqlNc46A_HwF-r7w?e=9mdhWK) (`on-going`). 


<details> 
<summary>Preparing data on your own (click for details)</summary>

1. Preprocessing corpora:
   ```
   python pretreatment/prepare_corpora.py --dataset MSRVTT --sort_vocab --attribute_first
   python pretreatment/prepare_corpora.py --dataset VATEX --sort_vocab --attribute_first
   ```
2. Feature extraction:


* Downloading all video files of MSRVTT and VATEX
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
* Extracting motion features: refer to [yangbang18/video-classification-3d-cnn](https://github.com/yangbang18/video-classification-3d-cnn)

* Extracting audio features: refer to [yangbang18/vggish](https://github.com/yangbang18/vggish)
</details>

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

**Notes:** In the publication version of our paper, attribute prediction (AP) is renamed as concept detection (CD). Here I keep the task name unchanged because of laziness.

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
@inproceedings{yang2022clip,
  title={CLIP Meets Video Captioning: Concept-Aware Representation Learning Does Matter},
  author={Yang, Bang and Zhang, Tong and Zou, Yuexian},
  booktitle={Pattern Recognition and Computer Vision: 5th Chinese Conference, PRCV 2022, Shenzhen, China, November 4--7, 2022, Proceedings, Part I},
  pages={368--381},
  year={2022},
  organization={Springer}
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
