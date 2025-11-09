


# EISAI Anime Interpolator

<div align="center">

![](./supplementary/teaser.png)

**Improving the Perceptual Quality of 2D Animation Interpolation**

[Shuhong Chen](https://shuhongchen.github.io/) · [Matthias Zwicker](https://www.cs.umd.edu/~zwicker/)

*ECCV 2022*

[![arXiv](https://img.shields.io/badge/arXiv-2111.12792-b31b1b.svg)](https://arxiv.org/abs/2111.12792)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/ShuhongChen/eisai-anime-interpolator)
[![Poster](https://img.shields.io/badge/ECCV-Poster-blue)](./eccv2022_eisai_poster.pdf)
[![Video](https://img.shields.io/badge/YouTube-Video-red?logo=youtube)](https://youtu.be/jy4HKnG9YA0)
[![Colab](https://img.shields.io/badge/Colab-Demo-orange?logo=google-colab)](https://colab.research.google.com/github/ShuhongChen/eisai-anime-interpolator/blob/master/_notebooks/eisai_colab_demo.ipynb)

</div>

---

### 🚩 Note on this Version

> **Note:** This is a fork of the original project [EISAI Anime Interpolator](https://github.com/ShuhongChen/eisai-anime-interpolator).
>
> This version is maintained primarily for **testing, bug fixes, and dependency updates**.
>
> All core research and academic contributions belong to the original authors: **[Shuhong Chen](https://shuhongchen.github.io/) and [Matthias Zwicker](https://www.cs.umd.edu/~zwicker/)**.

---

## 📖 Abstract

Traditional 2D animation is labor-intensive, often requiring animators to manually draw twelve illustrations per second of movement. While automatic frame interpolation may ease this burden, 2D animation poses additional difficulties compared to photorealistic video. In this work, we address challenges unexplored in previous animation interpolation systems, with a focus on improving perceptual quality.

**Key Contributions:**
- 🎯 **SoftsplatLite (SSL)**: A forward-warping interpolation architecture with fewer trainable parameters and better perceptual performance
- 📏 **Distance Transform Module (DTM)**: Leverages line proximity cues to correct aberrations in difficult solid-color regions
- 📊 **RRLD Metric**: Restricted Relative Linear Discrepancy metric to automate the previously manual training data collection process
- 👥 **User Study**: Establishes that LPIPS perceptual metric and chamfer line distance (CD) are more appropriate measures of quality than PSNR and SSIM

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Colab Demo](#-colab-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Docker Setup](#-docker-setup-advanced)
- [Evaluation](#-evaluation)
- [Training](#-training)
- [Downloads](#-downloads)
- [Citation](#-citation)

---

## 🚀 Quick Start

The fastest way to try the model:

**Option 1: Colab** (No setup required)  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShuhongChen/eisai-anime-interpolator/blob/master/_notebooks/eisai_colab_demo.ipynb)

**Option 2: Local Installation** (Conda)  
See [Installation](#-installation) section below.

---

## 🎮 Colab Demo

Try the model in your browser without any installation! The notebook sets up the environment and runs the interpolation script automatically.

👉 [Open eisai_colab_demo.ipynb](https://colab.research.google.com/github/ShuhongChen/eisai-anime-interpolator/blob/master/_notebooks/eisai_colab_demo.ipynb)
---

## 📥 Downloads

All downloads can be found in the Google Drive folder: [eccv2022_eisai_anime_interpolator_release](https://drive.google.com/drive/folders/1AiZVgGej7Tpn95ats6967neIEPdShxWy?usp=sharing)

### Required for Inference

- **`checkpoints.zip`** - Extract to the root project directory
  - Contains `ssl.pt`, `dtm.pt`, and `anime_interp_full.ckpt` (pretrained model from [AnimeInterp](https://github.com/lisiyao21/AnimeInterp))
  - These checkpoints are all you need for inference

### Optional for Evaluation

- **ATD12k dataset** - Download from [AnimeInterp](https://github.com/lisiyao21/AnimeInterp)
- **`rfr_540p.zip`** - Our repacked flows (SGM+RFR)
  
<details>
<summary>📁 Expected file structure for evaluation</summary>

```
_data/atd12k/
    raw/
        test_2k_540p/          # Raw image data from AnimeInterp
            Disney_ ...
            Japan_ ...
    preprocessed/
        rfr_540p/              # Precomputed flow pickles
            test/
                Disney_ ...
                Japan_ ...
```

> **Note:** Our repacked flows (SGM+RFR) differ from the AnimeInterp download (SGM only). The full SGM+RFR flows are [complicated to extract](https://github.com/lisiyao21/AnimeInterp/blob/b38358335fcd7361a199c1f7d899d457724ecee0/test_anime_sequence_one_by_one.py#L127) from their repo, so we provide them for convenience.

</details>

### Optional for Full User Study

- **`user_study_full.zip`** - Extract and open `index.html` in a browser supporting WEBP animations

> While we can't release the new training data collected in this work, our specific sources are listed in the paper supplementary. Our RRLD data collection pipeline (below) allows you to recreate our dataset or assemble your own from source animations.

---

## 💻 Installation

### Prerequisites

- Tested on: **Ubuntu 22.04 LTS** (other Linux distributions may work)
- Recommended GPU: **NVIDIA RTX 3090** (with up-to-date NVIDIA drivers and CUDA)
- Conda (Miniconda or Anaconda) — we recommend using conda environments
- Python 3.x

### Tested configuration

This repository has been verified to run successfully with the following setup:

- OS: Ubuntu 22.04 LTS
- GPU: NVIDIA RTX 3090 (CUDA-compatible drivers)
- Environment: Conda environment created from `environment.yml` and activated (`conda activate eisai_env`)

If you use a different configuration, you may still be able to run the code but results and exact package versions may vary.

### Step 1: Install Miniconda

If you don't have Miniconda installed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Clone Repository

```bash
git clone https://github.com/ShuhongChen/eisai-anime-interpolator.git
cd eisai-anime-interpolator
```

### Step 3: Create Environment

```bash
conda env create -f environment.yml
conda activate eisai_env
```

### Step 4: Download Checkpoints

Download `checkpoints.zip` from the [Google Drive folder](https://drive.google.com/drive/folders/1AiZVgGej7Tpn95ats6967neIEPdShxWy?usp=sharing) and extract it to the root project directory.

---

## 🎬 Usage

### Basic Frame Interpolation

Interpolate frames between two images:

```bash
mkdir -p ./temp/interpolate_demo_output

python3 -m _scripts.interpolate \
    ./demo_input/frame_001.png \
    ./demo_input/frame_002.png \
    --fps=12 \
    --out=./temp/interpolate_demo_output
```

**Parameters:**
- `--fps`: Number of frames per second for interpolation (controls how many intermediate frames are generated)
- `--out`: Output directory for interpolated frames

> **Note:** This script uses RFR/RAFT flows without SGM initialization. See our paper supplementary section on SGM computation tradeoffs. Due to package version differences, RFR/RAFT flows may differ slightly from the original AnimeInterp repo.

### Custom Images

```bash
python3 -m _scripts.interpolate \
    /path/to/frame0.png \
    /path/to/frame1.png \
    --fps=12 \
    --out=./temp/output
```

---

## 🐋 Docker Setup (Advanced)

For users who prefer Docker (requires GPU support):

### Step 1: Configure Environment

Make a copy of the machine config template:

```bash
cp ./_env/machine_config.bashrc.template ./_env/machine_config.bashrc
```

Edit `./_env/machine_config.bashrc` and set `$PROJECT_DN` to the absolute path of this repository folder. Other variables are optional.

### Step 2: Pull or Build Docker Image

**Option A: Pull pre-built image**
```bash
make/docker_pull
make/shell_docker
```

**Option B: Build image yourself**
```bash
make/docker_build
make/shell_docker
```

> **Note:** These are bash scripts in the `./make` folder, not `make` commands.

---

## 📊 Evaluation

---

## 📊 Evaluation

Reproduce the best-result metrics on ATD12k from our paper:

```bash
python3 -m _scripts.evaluate
```

### Expected Results

Output should match up to precision differences (tested on GTX1080ti):

| Subset | Metric  | Score      |
|--------|---------|------------|
| all    | lpips   | 3.4943E-02 |
| all    | chamfer | 4.3505E-05 |
| all    | psnr    | 29.29      |
| all    | ssim    | 95.15      |
| east   | lpips   | 3.8260E-02 |
| east   | chamfer | 4.9791E-05 |
| west   | lpips   | 3.2915E-02 |
| west   | chamfer | 3.9660E-05 |

---

## 🎓 Training

### RRLD Data Pipeline

Extract training data from a source video using RRLD. This all-in-one script performs:
- Re-encoding
- Deduplication
- RRLD filtering
- Triplet image extraction
- Triplet flow estimation

```bash
bash ./_scripts/rrld_pipeline.sh \
    /path/to/video.mp4 \
    ./temp/rrld_demo_output
```

> **Note:** The flows here use RFR/RAFT instead of [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) to simplify the repo. As discussed in the paper, RRLD is quite robust to choice of flow estimator. Unlike the paper, this script doesn't use [TransNetV2](https://github.com/soCzech/TransNetV2) to limit one sample per cut; this restriction can be reimposed by filtering `rrld.txt` before image extraction.

### Train Model

Train on RRLD-extracted data:

```bash
python3 -m _train.frame_interpolation.train \
    ./temp/rrld_demo_output \
    ./temp/training_demo_output
```

> **Note:** This starter code uses the same hyperparameters as our best-performing experiment.

---

## 📄 Citation

If you use our work in your research, please cite our paper:

```bibtex
@inproceedings{chen2022eisai,
    title={Improving the Perceptual Quality of 2D Animation Interpolation},
    author={Chen, Shuhong and Zwicker, Matthias},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2022}
}
```

---

## 📜 License

See [license.txt](./license.txt) for details.

---

## 🙏 Acknowledgments

This project builds upon [AnimeInterp](https://github.com/lisiyao21/AnimeInterp). We thank the authors for their foundational work.

---

<div align="center">

**[⬆ Back to Top](#eisai-anime-interpolator)**

Made with ❤️ for the animation community

</div>






