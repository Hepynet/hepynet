#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pdnn_lfv",
    version="0.3.0",
    author="Zhe Yang",
    author_email="starprecursor@gmail.com",
    description="A package for pDNN study in LFV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StarPrecursor/pdnn-lfv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires=">=3.7",
    python_requires=">=3.5,<3.9",
    install_requires=[
        # "tensorflow==2.2.0",
        # "keras==2.3.1",
        # "numpy",
        # "matplotlib",
        # "scikit-learn",
        # "configparser",
        # "reportlab",
        # "pandas",
        # "seaborn",
        # "hyperopt",
        # "eli5",
    ],
    entry_points={
        "console_scripts": ["hepynet=lfv_pdnn.main.execute:execute",],
    },
)
