# pDNN Code for LFV study v1.0 (pDNN_LFV)

![forthebadge](https://img.shields.io/badge/pdnn__lfv-v1.0-blue)
![forthebadge](https://img.shields.io/badge/status-developing-yellow)


The codes contains modules to use lfv ntuples of newest round and perform pDNN training.

* pDNN_LFV can produce array from the ntuple for the new round ntuples used in LFV group.
* pDNN_LFV contains modules which handle model training process like model configuration, performance report and so on.
* Examples are given to demonstrate the usage of pDNN_LFV to generate array and perform a training.

About pDNN for LFV:

* pDNN means parameterized deep neural network. It use a traditional set of event-level features (neural networks or other multi-variable methods) plus one or more feature(s) that describe the larger scope of the problem such as a new particle mass.
* One or a few training for the whole mass region.
* The method has already being used/under development for several analyses, for example: low-mass Z', high mass H4l, di-Higgs->llbb&nu;&nu; and etc.
* A related paper can be found [here](https://arxiv.org/pdf/1601.07913.pdf)

## Environment

### Method 1 - Use Docker (recommended)
Install [Docker](https://www.docker.com/) if not installed.  
Set up docker image:
```shell
cd docker
source build_docker.sh
```
On every startup:  
first cd to git repository's base directory, then
```bash
source docker/start_docker.sh
```

### Method 2 manually set environment with conda
#### Install Used Softwares
1. Install [python 3.7+](https://www.python.org/downloads/windows/)
2. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
3. Install [Git](https://git-scm.com/downloads)

#### Install Python Packages
Open a **Anaconda powershell Prompt** (powshell in VSCode is **NOT RECOMMENDED**) to create conda environment and activate.
```shell
> conda create -n pdnn python=3.7
> conda activate pdnn
```
Then install python packages (you can run following commands in Git Bash):  
* **numpy** (use -n pdnn to only install package for specified environment):
```shell
> conda install -n pdnn numpy
```
* **matplotlib** (for plots)
```shell
> conda install -n pdnn matplotlib
```
* **keras** with **tensorflow** backend (for DNN training)  
First install tensorflow in conda. If you have a GPU supporting [CUDA](https://developer.nvidia.com/cuda-zone), following instructions to install [tensorflow-gpu](https://www.tensorflow.org/install/gpu). Otherwise, to install the tensorflow 2.0 (currently not available in conda), use pip for installation:
```shell
> pip install --upgrade pip
> pip install tensorflow
```
&nbsp; &nbsp; &nbsp; &nbsp; Or use conda (not recommended)
```shell
> conda install -n pdnn tensorflow
```
&nbsp; &nbsp; &nbsp; &nbsp; To install newest version(recommended) of keras, use pip:
```shell
> pip install keras
```
&nbsp; &nbsp; &nbsp; &nbsp; Or with conda (newest version not available, not recommended).
```shell
> conda install -n pdnn keras
```
* **sklearn**
```shell
> conda install -n pdnn scikit-learn
```
* **eli5**
```shell
> conda install -n pdnn -c conda-forge eli5
```
* **ConfigParser**
```shell
> conda install -n pdnn ConfigParser
```
* **Reportlab**
```shell
> conda install -n pdnn reportlab
```
* **root** (optional unless need to produce array from ntuple)  
(not available on Windows currently)
```shell
> conda install -n pdnn -c conda-forge root
```
* **pandas** (optional)
```shell
> conda install -n pdnn pandas
```
* **jupyter lab** (optional)
```shell
> conda install -n pdnn jupyterlab
```
#### Fist time run
On main folder, where setup.py exists:
```shell
> pip install -e .
```
