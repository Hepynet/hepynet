# pDNN Code for LFV study v1.0 (pDNN_LFV)

![forthebadge](https://img.shields.io/badge/pdnn__lfv-v0.3.0-blue)
![forthebadge](https://img.shields.io/badge/status-developing-green)

The codes contains modules to use lfv ntuples of newest round and perform pDNN training.

- pDNN_LFV use a simple configuration file to specify the input ntuples, model configurations, outputs and so on.
- The corresponding modules cas also be used in an independent python script or Jupyter lab.
- Examples are given to demonstrate the usage of pDNN_LFV to generate array and perform a training.
- pDNN_LFV was developed for lepton flavor violation study, but can be used for different analysis. Further tests and developments are still needed. Currently, the framework is tested also on low mass Z prime study except LFV study.

About pDNN study for LFV:

- pDNN means parameterized deep neural network. It use a traditional set of event-level features (neural networks or other multi-variable methods) plus one or more feature(s) that describe the larger scope of the problem such as a new particle mass.
- One or a few training for the whole mass region.
- The method has already being used/under development for several analyses, for example: low-mass Z', high mass H4l, HH->llbb&nu;&nu; and etc.
- A related paper can be found [here](https://arxiv.org/pdf/1601.07913.pdf)

## **Environment**

### **Method 1** - Use Docker (recommended)

Install [Docker](https://www.docker.com/) if not installed.  
Set up docker image:

```shell
cd docker
source build_docker.sh
```

On every startup:  
first cd to git repository's base directory, then (in Linux)

```bash
source docker/start_docker.sh
```

or (in Windows PowerShell)

```bash
. docker/start_docker.bat
```

### **Method 2** manually set environment with conda

#### Install Used Softwares

1. Install [python 3.7+](https://www.python.org/downloads/windows/)
2. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
3. Install [Git](https://git-scm.com/downloads)

#### Install Python Packages

Open a **Anaconda powershell Prompt** (powshell in VSCode is **NOT RECOMMENDED**) to create conda environment and activate.

```shell
conda create -n pdnn python=3.7
conda activate pdnn
```

Then install python packages use conda or pip(use -n pdnn to only install package for specified environment):

- **keras** with **tensorflow** backend (for DNN training)  
  First install tensorflow in conda. If you have a GPU supporting [CUDA](https://developer.nvidia.com/cuda-zone), following instructions to install [tensorflow-gpu](https://www.tensorflow.org/install/gpu).

  ```bash
  conda install -n pdnn tensorflow keras
  ```

- required: **numpy**, **matplotlib**, **sklearn**, **eli5**, **configparser**, **Reportlab**, **pandas**, **seaborn**, **hyperopt**, **root** (not available on Windows currently, considering add matplotlib plot option in the future)

  (part of the packages only available in conda-forge)

  ```bash
  conda install -n pdnn numpy matplotlib scikit-learn configparser reportlab pandas seaborn hyperopt
  conda install -n pdnn -c conda-forge root uproot eli5
  ```

- optional: **jupyter lab**

  ```bash
  conda install -n pdnn -c conda-forge jupyterlab
  ```

#### Fist time run

On main folder, where setup.py exists:

```bash
conda develop .
```

or (not recommended)

```bash
pip install -e .
```

If you are using the docker scripts "start_docker.bat/start_docker.sh", you can skip this step.