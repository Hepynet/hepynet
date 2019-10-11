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

## Requirements (work on Windows 10)

### Install Used Softwares
1. Install [python 3.7+](https://www.python.org/downloads/windows/)
2. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
3. Install [Git](https://git-scm.com/downloads)
4. Install [VsCode](https://code.visualstudio.com/docs/setup/windows)

### Install Python Packages
1. Install python packages(you can run following commands in Git Bash): 
    Open a Anaconda powershell Prompt (powshell in VsCodecreate is not recommended) to create conda environment and activate.
    ```
    > conda create -n pdnn python=3.7
    > conda activate pdnn
    ```
    * numpy:
    ```
    > conda install numpy
    ```
    * matplotlib (for plots)
    ```
    > conda install matplotlib
    ```
    * keras with tensorflow backend (for DNN training)  
    First install tensorflow in conda. If you have a GPU supporting [CUDA](https://developer.nvidia.com/cuda-zone), following instructions to install [tensorflow-gpu](https://www.tensorflow.org/install/gpu). Otherwise:
    ```
    > conda install tensorflow
    ```
    Then install keras (with tensorflow backend).
    ```
    > conda install keras
    ```
    * sklearn
    ```
    > conda install scikit-learn
    ```
    * pandas (optional)
    ```
    > conda install pandas
    ```
    * jupyter lab
    ```
    > conda install jupyterlab
    ```
    
### System:
    * Work on Linux, Windows and WSL([Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)). If run on Win10 OS, the make_array module and train module can't work together due the conflct of x86/x64 instruction set conflicts.
    * Hardware need to support tensorflow. (better training speed with an tensorflow supported dedicated graphics/calcution cards)

## Usage
(Will be updated with test codes and examples.)
```python
pass
```
