# **Hepynet**

**H**igh **e**nergy physics, **py**thon-based, neural-**net**work assistant framework

![forthebadge](https://img.shields.io/badge/hepynet-v0.4.0-blue)

## **Introduction**

Goal of the hepynet: perform DNN related high energy physics analysis tasks with simple config files

- for **ATLAS Analysis**: include supports for various ATLAS analysis jobs

- **Config Driven**: all tasks defined by a simple config file

- **Python Based**: codes are written in Python, which is the mainstream language for DNN studies

## **Setup**

### Clone Git repository

```bash
git clone --recursive git@github.com:StarPrecursor/hepynet.git
```

### Installation

- **Method 1** - Use Docker

  Install [Docker](https://www.docker.com/) if not installed.

  On every startup

  - **Linux Command Line**

    ```bash
    source docker/start_docker.sh
    ```

  - **Windows PowerShell**

    ```bash
    . docker/start_docker.bat
    ```

  - **Windows file explorer**

    double-click docker/start_docker.bat

  Note: if the Docker image is not installed yet, this will automatically pull the required image from [Docker Hub](https://hub.docker.com/)

- **Method 2** - Use [Conda](https://www.anaconda.com/)

  ```bash
  conda create -n hepynet python=3.8
  conda activate hepynet
  cd src
  pip install .
  cd ..
  ```

### GPU support

- You can refer to [Tensorflow GPU support](https://www.tensorflow.org/install/gpu) to set up environment to use GPU for training

- This is not mandatory, CPU alone is enough to run hepynet

## **Preparations**

- **Prepare numpy arrays as inputs**

  - You can write your own code to generate numpy arrays from root files
  - Or refer to [hepynet_root_npy](https://github.com/HEPTools/hepynet_root_npy) for more information about **root <--> numpy transformation**
  - Numpy arrays should be organized as following

    ```bash
    Data_folder/path_to_array_folder/array_version/campaign/region/feature.npy
    ```

    Note:

    - "Data_folder" is what you set in pc_meta.yaml
    - "path_to_array_folder" is what you specified in train/apply job config files
    - each numpy array file should only save **one** input feature

- **Prepare job config files**

  all config files are put under share folder, there are 3 types of config files you should prepare/modify

  - **cross_platform.pc_meta.yaml**

    You should set up the data folder (where you keep the input numpy arrays) in this file

  - **train configs**: configs for a model training job

  - **apply configs**: configs for a model applying job

  please refer to [config_preparing.md](docs/config_preparing.md) for more details

## **Usage**

```bash
hepynet -h
usage: hepynet [-h] [-d] [-v] [yaml_configs [yaml_configs ...]]

positional arguments:
  yaml_configs

optional arguments:
  -h, --help     show this help message and exit
  -d, --debug    run in debug mode
  -v, --verbose  verbose debug infomation
```

## **Release note - v0.4.0**

- This is the first public release, totally re-written to simplify the workflow and optimize the memory usage.

- Part of the utilities (Bayesian optimization, k-fold training ... ) are disabled to reduce the test/debug work load, will be added back in the coming releases.
