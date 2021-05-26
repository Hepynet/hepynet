# **Hepynet**

**H**igh **e**nergy physics, **py**thon-based, neural-**net**work assistant framework

![forthebadge](https://img.shields.io/badge/hepynet-v0.4.2-blue)

## **Introduction**

The goal of the hepynet: perform DNN related high energy physics analysis tasks with simple config files

- for **ATLAS Analysis**: include supports for various ATLAS analysis jobs

- **Config Driven**: all tasks defined by a simple config file

- **Python Based**: codes are written in Python, which is the mainstream language for DNN studies

## **Installation**

```bash
pip install hepynet
```

### GPU support

- You can refer to [Tensorflow GPU support](https://www.tensorflow.org/install/gpu) to set up the environment to use GPU for training

- This is not mandatory, CPU alone is enough to run hepynet

## **Set Up the Workspace**

Please refer to [hepynet_example](https://github.com/Hepynet/hepynet_example) to see how to set up the workspace of hepynet.

## **Release Note - v0.4.2**

- Re-implement auto-tuning functions with [Ray](https://docs.ray.io/en/master/index.html) (currently, support three hyperparameter-tuning algorithms: [Ax](https://ax.dev/), [HyperOpt](http://hyperopt.github.io/hyperopt/) and [HEBO](https://github.com/huawei-noah/noah-research/tree/master/HEBO))
- Minor issues fixes and improvements
