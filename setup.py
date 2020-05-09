import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lfv_pdnn",
    version="1.0.1",
    auther="Zhe Yang",
    auther_email="starprecursor@gmail.com",
    description="A packge for pDNN study in LFV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StarPrecursor/pdnn-lfv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'execute_pdnn_jobs=share.execute:execute',
        ],
    },
)
