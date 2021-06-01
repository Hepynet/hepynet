# Procedures for Publishing New Release

## Update API documents with pdoc

- in hepynet root directory run

  ```bash
  pdoc --html --force -o docs\api hepynet
  ```

- [optionally] update in personal CERN homepage

## Format the code with black

- in hepynet root directory run

  ```bash
  black hepynet
  ```

- note the black configurations are defined in pyproject.toml

- more details about using black see: [black homepage](https://github.com/psf/black)

## Build and upload package to **testpypi**

- suggest to use Linux for package building/uploading

- see procedures on: [tutorials](https://packaging.python.org/tutorials/packaging-projects/)

- note: need to build and upload to testpypi first and upload to pypi after tests finished

## Test package installation and usage

- use a clean environment to preform the tests

- note: TestPyPI doesn’t have the same packages as the live PyPI, should use "--no-deps" for installation test

## Build & Test rc package to **pypi**

- temporarily add suffix "rc0" to the version in setup.cfg

- build and upload

- use a clean environment to preform the tests

## Build and upload package to **pypi**

- build and upload official release to pypi

## Add new release tag on Git
