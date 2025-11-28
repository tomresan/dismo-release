[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://compvis.github.io/dismo/)
[![Paper](https://img.shields.io/badge/arXiv-paper-b31b1b)](https://openreview.net/forum?id=jneVld5iZw)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/CompVis/dismo)
<h2 align="center"><i>DisMo</i>: Disentangled Motion Representations</h2>
<h2 align="center"><i>DisMo</i>: for Open-World Motion Transfer</h2>
<div align="center"> 
  <a href="https://www.linkedin.com/in/thomas-ressler-494758133/" target="_blank">Thomas Ressler-Antal</a> · 
  <a href="https://ffundel.de/" target="_blank">Frank Fundel</a><sup>*</sup> · 
  <a href="https://www.linkedin.com/in/malek-ben-alaya/" target="_blank">Malek Ben Alaya</a><sup>*</sup>
  <br>
  <a href="https://stefan-baumann.eu/" target="_blank">Stefan A. Baumann</a> · 
  <a href="https://www.linkedin.com/in/felixmkrause/" target="_blank">Felix Krause</a> · 
  <a href="https://www.linkedin.com/in/ming-gui-87b76a16b/" target="_blank">Ming Gui</a> · 
  <a href="https://ommer-lab.com/people/ommer/" target="_blank">Björn Ommer</a>
</div>
<p align="center"> 
  <b>CompVis @ LMU Munich, MCML</b>
  <br/>
  <i>* equal contribution</i>
  <br/>
  NeurIPS 2025 Spotlight
</p>


![DisMo learns abstract motion representations that enable open-world motion transer](docs/static/images/teaser.png)


## 📋 Overview
We present <b>DisMo</b>, a paradigm that learns a semantic motion representation space from videos that is disentangled from static content information such as appearance, structure, viewing angle and even object category. We leverage this invariance and condition off-the-shelf video models on extracted motion embeddings. This setup achieves state-of-the-art performance on open-world motion transfer with a high degree of transferability in cross-category and -viewpoint settings. Beyond that, DisMo's learned representations are suitable for downstream tasks such as zero-shot action classification.

## 🛠️ Setup
We have tested our setup on `Ubuntu 22.04.4 LTS`.

First, clone the repository into your desired location:
```
git clone git@github.com:CompVis/dismo.git
cd dismo
```

We recommend using a package manager, <i>e.g.,</i> [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). When installed, you can create and activate a new environment:
```
conda create -n dismo python=3.11
conda activate dismo
```

Afterwards install PyTorch. We have tested this setup with `PyTorch 2.7.1` and `CUDA 12.6`:
```
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```
If you need to install an alternative version, <i>e.g.</i> due to incompatible CUDA versions, see the [official instructions](https://pytorch.org/get-started/locally/).

Finally, install all other packages:
```
pip install -r requirements.txt
```

<i>(Optional)</i> We use the [torchcodec](https://github.com/meta-pytorch/torchcodec) package for data loading, which expects `ffmpeg` to be installed. If you plan to train DisMo yourself and you don't have a ffmpeg version installed yet, an easy way is to use `conda`:
```
conda install ffmpeg
```

## 🚀 Usage
To use DisMo for motion transfer purposes, we provide the code and LoRA weights of an adapted CogVideoX-5B-I2V video model,conditioned on motion embeddings and text prompts. The simplest way to use it is via `torch.hub`:
```
cogvideox = torch.hub.load("CompVis/DisMo", "cogvideox5b_i2v_large")
```

Alternatively, you can also instantiate and load the model yourself:
```
from dismo.video_model_finetuning.cogvideox import CogVideoXMotionAdapter_5B_TI2V_Large
cogvideox = CogVideoXMotionAdapter_5B_TI2V_Large()
state_dict = torch.load("/path/to/finetuned/cogvideox/checkpoint/cogvideox5b_i2v_large.pt")
cogvideox.load_state_dict(state_dict, strict=False)
cogvideox.requires_grad_(False)
cogvideox.eval()
```

You can then use the model's `sample` function to generate new videos by transferring motion from `motion_videos` to `images`. Since CogVideoX is a text-to-video model at its core, we recommend to additionally provide describing `prompts` alongside the target images for better generation results:
```
generated_videos = cogvideox.sample(
    motion_videos=driving_videos,
    images=target_images,
    prompts=target_text_prompts,
)
```
The sample function comes with some other arguments (e.g., classifier-free text guidance). Please have a look in the code for more details.

### Motion Extraction
During motion transfer, the video model internally uses DisMo's pre-trained motion extractor for encoding input videos into motion embeddings. However, the motion extractor can also be used as a standalone model to extract sequences of motion embeddings from input videos. This might be useful for video analysis purposes or other downstream tasks. Once again, the easiest way to load the model is via `torch.hub`:
```
motion_extractor = torch.hub.load("CompVis/DisMo", "motion_extractor_large")
```

Similarly, you can also manually instantiate and load the model:
```
from dismo.model import MotionExtractor_Large
motion_extractor = MotionExtractor_Large()
state_dict = torch.load("/path/to/motion/extractor/checkpoint/motion_extractor_large.pt")
motion_extractor.load_state_dict(state_dict)
motion_extractor.requires_grad_(False)
motion_extractor.eval()
```

To extract motion sequences from arbitrarily long videos, we provide the `forward_sliding` function, which extracts embeddings consecutively in a sliding window fashion. This is necessary, since DisMo only saw video clips of length 8 during training:
```
import torch

# videos are expected to have shape [B, T, H, W, C] in (-1, 1) range
dummy_video = torch.rand((B, num_frames, 256, 256, 3)).mul(2).sub(1)

# we get a motion embedding for each frame, except for the last 4
motion_embeddings = motion_extractor.forward_sliding(dummy_video)
```
Note that the resulting motion embeddings have a temporal length of `num_frames - 4`, since the longest possible prediction distance was set to 4 during training.


## 🔥 Training
If you want to train DisMo yourself, we provide a training script that is suitable for multi-gpu training. Please note that the script instantiates DisMo  with default parameters. To train other variants (e.g., changing the width, depth, etc.) you must modify the `train.py` accordingly. This equally holds true for video model adaptation.

### Data Preparation
DisMo needs unlabelled videos for training. This repository takes advantage of the [webdataset](https://github.com/webdataset/webdataset) library and format for efficient and scalable data loading. Please refer to their page for further instructions on how to shard your video files accordingly.

### Launching Training
Single-GPU training can be launched via
```shell
python train.py --data_paths /path/to/preprocessed/shards --out_dir output/test --compile True
```
Similarly, multi-GPU training, e.g., on 2 GPUs, can be launched using torchrun:
```shell
torchrun --nnodes 1 --nproc-per-node 2 train.py [...]
```
Training can be continued from a previous checkpoint by specifying, e.g., `--load_checkpoint output/test/checkpoints/checkpoint_0100000.pt`.
Remove `--compile True` for significantly faster startup time at the cost of slower training & significantly increased VRAM usage.


## 🤖 Models
We release the weights of our pre-trained motion extractor and LoRA weight of an adapted CogVideoX-5B-I2V model via huggingface at https://huggingface.co/CompVis (under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) license). We will release other model variants in the future, e.g., more recent and powerful video models, from which DisMo might benefit from. Due to legal concerns, we do not release the weights of the frame generator that was trained alongside the motion extractor.

## Code Credit
- Some code is adapted from [flow-poke-transformer](https://github.com/CompVis/flow-poke-transformer) by Stefan A. Baumann et al. (LMU), which in turn adapts some code from  [k-diffusion](https://github.com/crowsonkb/k-diffusion) by Katherine Crowson (MIT)
- The code for fine-tuning CogVideoX models is adapted from [CogKit](https://github.com/THUDM/CogKit) (Apache 2.0)
- The DINOv2 code is adapted from [minDinoV2](https://github.com/cloneofsimo/minDinoV2) by Simo Ryu, which is based on the [official implementation](https://github.com/facebookresearch/dinov2/) by Oquab et al. (Apache 2.0)

## 🎓 Citation
If you find our work useful, please cite our paper:
```bibtex
@inproceedings{resslerdismo,
  title={DisMo: Disentangled Motion Representations for Open-World Motion Transfer},
  author={Ressler-Antal, Thomas and Fundel, Frank and Alaya, Malek Ben and Baumann, Stefan Andreas and Krause, Felix and Gui, Ming and Ommer, Bj{\"o}rn},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
