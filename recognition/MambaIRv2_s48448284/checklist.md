Days 1–2 — Setup & Data

[x] Fork the repo, checkout topic-recognition, create your folder recognition/MambaIRv2_sXXXXXXX/

[x] Find the starter notebook mentioned in the spec (check Blackboard/course materials)
- https://colab.research.google.com/drive/1AyC5c4tVwstI0irzbgbVxQecxg1D_ux7?usp=sharing 

[x] Set up your conda environment locally with PyTorch

[x] Understand the dataset — the spec says “supplied test images” so check what’s actually provided

[X] Write your feasibility review (needed for tutor check-off AND README)

[X] Start dataset.py

Days 3–4 wed-thur — Baseline

[x] Get a CNN baseline working


[X] Get the pretrained MambaIRv2 weights running (frozen, no fine-tuning)
[X] Run inference on the test set — this becomes your baseline
[X] Record PSNR, SSIM, LPIPS, Colorfulness scores on the baseline
[X] Commit this working baseline

Days 5–7 fri-sun— Fine-tuning

[X] Implement fine-tuning in train.py
[X] Train locally on your RTX 5060
[] Compare metrics before/after fine-tuning
[] If you need more compute, push to Rangpur via sbatch

Days 8–9 — Analysis & Predict

[X] Write predict.py with side-by-side visualisations (greyscale input / baseline / fine-tuned / ground truth)
[] Pick 3–5 failure cases and write the autopsy (color bleeding, desaturation, chromatic hallucinations)
[] Draft your engineering recommendation

Days 10–11 — README & Submit

[] Complete README with all required sections
[] Final commit sweep
[] Open Pull Request, Turnitin PDF submission


## Installation

``` 
git clone https://github.com/amyntyalien/PatternAnalysis-2026.git
cd PatternAnalysis-2026

conda create -n mamba-colour python=3.10 pip -y
conda activate mamba-colour

# 5. Install dependencies
pip install --no-cache-dir torch torchvision transformers==4.44.2 \
    einops ninja packaging matplotlib opencv numpy \
    scikit-image lpips timm==0.9.16 basicsr scikit-learn path

# 6. Install mamba_ssm from pre-built wheel (no compilation needed!)
pip install --no-cache-dir \
    "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

```

# Environment Set Up
**1. Windows prerequisites**
In an Administrator PowerShell, install Ubuntu if needed:
```wsl --install -d Ubuntu-24.04```
Install the current NVIDIA driver for Windows, then restart Windows and open Ubuntu. Do not install a Linux NVIDIA driver inside WSL.

**2. Install CUDA tools in Ubuntu**
```
sudo apt update
sudo apt install -y build-essential git wget python3.12-venv

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc

nvidia-smi
nvcc --version
```

**3. Create the Python environment and install dependencies**
```
mkdir -p ~/.venvs
/usr/bin/python3.12 -m venv ~/.venvs/comp3710
source ~/.venvs/comp3710/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.14.1" "causal-conv1d==1.7.0" "transformers==4.44.2" einops ninja packaging
```
Check that PyTorch sees this GPU
```python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"```

**4. Build the compatible Mamba release**
Mamba 2.2.6.post3 is the version we tested. Its build needs C++20 with this PyTorch version, so adjust the local source before installing:
```
cd ~
git clone --depth 1 --branch v2.2.6.post3 https://github.com/state-spaces/mamba.git mamba-226
cd ~/mamba-226

sed -i 's/-std=c++17/-std=c++20/g' setup.py

MAX_JOBS=1 MAMBA_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0 \
python -m pip install --no-build-isolation --no-cache-dir --no-deps .
```

**5. Verify Mamba on the GPU**
``` python -c "import torch, mamba_ssm; from mamba_ssm import Mamba; x=torch.randn(2,64,16,device='cuda'); m=Mamba(d_model=16,d_state=16,d_conv=4,expand=2).to('cuda'); y=m(x); torch.cuda.synchronize(); print(mamba_ssm.__version__, tuple(y.shape), torch.cuda.get_device_name(0))"```

**Normal project work**
Open Ubuntu and activate the environment with source ~/.venvs/comp3710/bin/activate

```
wsl -d Ubuntu-24.04
cd /mnt/c/Users/amy_c/Documents/UNI/COMP3710-Project/PatternAnalysis-2026
source ~/.venvs/comp3710/bin/activate
```

In VS Code, connect to WSL: Ubuntu-24.04 and select this Python interpreter:
```/home/amy_c/.venvs/comp3710/bin/python```
(in ubuntu run:)
```cd /mnt/c/Users/amy_c/Documents/UNI/COMP3710-Project/PatternAnalysis-2026
code .```


Verified versions in this environment:
- Python 3.12
- PyTorch 2.14.1+cu130
- causal-conv1d 1.7.0
- mamba-ssm 2.2.6.post3
- Transformers 4.44.2
Mamba needed a local C++20 build for compatibility with this PyTorch version. The adjusted source checkout is at ~/mamba-226. You shouldn’t need to rebuild it for normal project work.

Quick GPU check:
```python -c "import torch; from mamba_ssm import Mamba; x=torch.randn(2,64,16,device='cuda'); m=Mamba(d_model=16,d_state=16,d_conv=4,expand=2).to('cuda'); print(m(x).shape, torch.cuda.get_device_name(0))"```

It should print a (2, 64, 16) output and your RTX 5060.