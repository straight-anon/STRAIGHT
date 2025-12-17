# Setup

#### Activating openmmlab:
```
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
```

# Installing Dependencies
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n openmmlab python=3.8 -y
conda activate openmmlab

# install pytorch, cuda version can be adjusted based on GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# install OpenMMLab core stack
pip install -U openmim
mim install mmengine
pip install "mmcv==2.1.0"
pip install "mmdet==3.2.0"
pip install "mmpose==1.3.2"

# (Optional) Verify
python -c "import torch; import mmcv, mmdet, mmpose; print(torch.__version__, mmcv.__version__, mmdet.__version__, mmpose.__version__)"

git clone https://github.com/open-mmlab/mmpose.git

cd mmpose

wget https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth

wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth
```