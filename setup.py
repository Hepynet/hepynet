#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hepynet",
    version="0.4.0",
    author="Zhe Yang",
    author_email="starprecursor@gmail.com",
    description="High energy physics, python based, neural network framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StarPrecursor/hepynet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8,<3.9",
    install_requires=[
        "tensorflow==2.4.0",
        "keras==2.3.1",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "seaborn",
        "hyperopt",
        "eli5",
    ],
    entry_points={"console_scripts": ["hepynet=hepynet.main.execute:execute"]},
)
