# **Hepynet**

**H**igh **e**nergy physics, **py**thon-based, neural-**net**work assistant framework

![forthebadge](https://img.shields.io/badge/hepynet-v0.4.0-blue)

## **Introduction**

Goal of the hepynet: perform DNN related high energy physics analysis tasks with simple config files

- for **ATLAS Analysis**: include supports for various ATLAS analysis jobs

- **Config Driven**: all tasks defined by a simple config file

- **Python Based**: codes are written in Python, which is the mainstream language for DNN studies

## **Installation**

```bash
pip install hepynet
```

### GPU support

- You can refer to [Tensorflow GPU support](https://www.tensorflow.org/install/gpu) to set up environment to use GPU for training

- This is not mandatory, CPU alone is enough to run hepynet

## **Set Up the Workspace**

Please refer to [hepynet_workspace](https://github.com/StarPrecursor/hepynet_workspace) to see how to set up workspace of hepynet.

## **Release Note - v0.4.0**

- This is the first public release, totally re-written to simplify the workflow and optimize the memory usage.

- Part of the utilities (Bayesian optimization, k-fold training ... ) are disabled to reduce the test/debug work load, will be added back in the coming releases.
