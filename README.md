
---

> ⚠️ **The pre-trained weights and comparison results will be released after the paper is accepted.**

---

## 🛠️ Installation

### Step 1: Install PyTorch

Install PyTorch 2.0+ with CUDA 11.8 support:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Note**: Ensure your NVIDIA driver is compatible with CUDA 11.8. You can verify with `nvidia-smi`.

### Step 2: Install Python Dependencies

Install required Python packages:

```bash
# Image processing and visualization
pip install matplotlib scikit-learn scikit-image opencv-python

# Deep learning utilities
pip install einops timm==0.4.12 lpips thop

# Training and evaluation tools
pip install yacs joblib natsort h5py tqdm tensorboard tensorboardX

# Additional dependencies
pip install gdown addict future lmdb numpy pyyaml requests scipy yapf

# Optional: For advanced features
pip install packaging triton pytest chardet termcolor submitit fvcore seaborn
```

### Step 3: Install BasicSR and Custom Kernels

Install the project in development mode:

```bash
# Install selective scan kernels (required for Mamba modules)
cd kernels/selective_scan && pip install .
cd ../../../..

# Install BasicSR in development mode
python setup.py develop --no_cuda_ext
```

## 🚀 Training

### Training Overview


### Single GPU Training


```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py --opt Options/LOL_Deraining.yml
```

### Multi-GPU Training


```bash
bash train_multigpu.sh
```


## 📊 Dataset

The constructed dataset for this project can be downloaded from [here](https://pan.baidu.com/s/1DZ1JKD6MqlXcq05QBLaPGQ?pwd=erut)


## 📈 Results

Comparison results of different methods:

| Method | Baidu Netdisk |
|--------|-------------------|
| Our | [Results](https://pan.baidu.com/s/1nJ06mmfcyt2K2Wu9hDMO0g?pwd=883v) |

##  Acknowledgement

This project is built upon the following excellent works:

- [BasicSR](https://github.com/XPixelGroup/BasicSR): Open source image and video restoration toolbox
- [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer): One-stage Retinex-based Transformer for low-light enhancement

We thank the authors for their outstanding contributions to the community.


## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
