## Prerequisites and Installation

Step 0. Download and install [Anaconda](https://www.anaconda.com/products/individual) (Python 3.8 or higher) for your operating system.

Step 1. Create a conda environment and activate it:

```bash
conda create -n crack_gauge python=3.8 -y
conda activate crack_gauge
```

Step 2. Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/).

For GPU support, install the appropriate CUDA version of PyTorch. For example, for CUDA 11.6:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Step 3. Install OpenMMLab dependencies:

Step 3-1. Install OpenMMLab installer, [MIM](https://github.com/open-mmlab/mim):

```bash
pip install openmim
```

Step 3-2. Install OpenMMLab dependencies required for your application. For the full usage of Crack-Gauge install following dependencies:

```bash
mim install mmengine
mim install mmcv
mim install mmpretrain
mim install mmsegmentation 
mim install mmdet   
mim install mmagic 
```

Step 4. Install from requirements.txt:

```bash
pip install -r requirements.txt
```

Step 4. Install Crack-Gauge:

```bash
python setup.py develop
```

## Additional dependencies for specific applications


1. Web Demo powered by Streamlit :

```bash
pip install streamlit
pip install streamlit-image-comparison
```


