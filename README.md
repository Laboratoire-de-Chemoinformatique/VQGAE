# VQGAE

This repository is official implementation of Vector Quantized Graph-based AutoEncoder (VQGAE).
This tool is in active development, therefore API can be changed.

## Installation

This tool depends on the pytorch, pytorch-geometric and pytorch-lightning packages. If you want to use GPU,
then you need to specify NVIDIA GPU driver version during installation.
Therefore, we provide to instructions to install the VQGAE package with CPU-only and GPU versions.

In case you don't have administrator (sudo) permissions on your machine, then usage of miniconda or mambaforge package manager is
highly advised.

### Prerequesties

The installation is based on a [Poetry package manager](https://python-poetry.org/docs/). It can be installed
directly on the machine or using conda package manager.

Here we specify installation with conda:

`conda create -n vqgae -c conda-forge "python<3.11" "poetry>1.4"`

When enviroment is created and activated (e.g. `conda activate vqgae`),
clone the repository of GTMtools from UniStra GitLab:

`git clone https://github.com/Laboratoire-de-Chemoinformatique/VQGAE.git`

go inside the folder `cd vqgae/`

and run `poetry install` to install the package and basic dependencies.

However, this is not the end. Next is pytorch installation and packages dependent on it.

### Pytorch GPU installation

First, check your GPU driver version with nvcc or nvidia-smi.

In case you haven't installed cudatoolkit drivers, and it requires administrator permissions that you don't have,
the only way to install is conda manager:

`conda install pytorch cudatoolkit=${CUDAVERSION} -c pytorch -c conda-forge -y`

where ${CUDAVERSION} is version of your GPU driver (the tool was tested with ${CUDAVERSION}=11.6).

In case you can manually install [cudatoolkit](https://developer.nvidia.com/cuda-toolkit),
pytorch can be installed as

`pip3 install torch --extra-index-url https://download.pytorch.org/whl/${CUDATORCH}`

where ${CUDATORCH} is CUDA version in pytorch format (cpu, cu102, cu113, cu116).

For more details, please visit the
[official pytorch installation documentation](https://pytorch.org/get-started/locally/)

Then, proceed with installation of [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html):

`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
-f https://data.pyg.org/whl/torch-${TORCH}+${CUDAPYG}.html`

where ${CUDATORCH} is CUDA version in pytorch format (cpu, cu102, cu113, cu116) and ${TORCH} is
version of installed pytorch (1.11.0, 1.12.0). For more details please, check the
[pytorch-geometric installation docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Finally, install pytorch-lightning and adabelief optimizer:

`pip install "pytorch-lightning>1.7.2" "adabelief-pytorch>=0.2.1"`

## Example usage

The tool work in command line mode. For the training you can simply run:

```bash
vqgae_train fit -c configs/vqgae_training.yaml
```

```bash
vqgae_encode predict -c configs/vqgae_encode.yaml
```

```bash
vqgae_decode predict -c configs/vqgae_decode.yaml
```


## Contributing

Contributions are welcome, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.

## Copyright

* [Tagir Akhmetshin ](tagirshin@gmail.com)
* [Arkadii Lin](arkadiyl18@gmail.com)
* [Alexandre Varnek](varnek@unistra.fr)