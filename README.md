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

## Requirements 

1. Install python packages: 
    * numpy, ROOT (for data manipulation)
    * matplotlib (for plots)
    * keras, sklearn, tensorflow (for DNN training)
2. System:
    * Work on Linux and WSL([Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10)). If run on Win10 OS, the make_array module and train module can't work together due the conflct of x86/x64 instruction set conflicts.
    * Hardware need to support tensorflow. (better training speed with an tensorflow supported dedicated graphics/calcution cards)

## Usage
(Will be updated with test codes and examples.)
```python
pass
```
