# Vector Quantized Graph-based AutoEncoder (VQGAE)

This repository is official implementation of Vector Quantized Graph-based AutoEncoder.

## :construction: Warning :construction:

This repository is under active development. Soon we will upload all models, weights, datasets etc.

### TO-DO tasks before the stable release:

- [x] First version of the code
- [x] Add ordering network
- [ ] Add inverse QSAR with genetic algorithm 
- [ ] Add links to datasets, weights and statistics
- [ ] Add trained models to HuggingFace
- [ ] Add tutorials in jupyter/colab
- [ ] Update documentation for VQGAE

## Installation

This tool depends on the pytorch, pytorch-geometric and pytorch-lightning packages. 
If you want to use GPU, then you need to manually specify NVIDIA GPU driver version during installation. 
Therefore, we provide to instructions to install the VQGAE package with conda and manually.

### Conda installation (recommended)

Here we specify installation with conda/mamba. 
First, you should copy reposotiry from git

```bash
git clone https://github.com/Laboratoire-de-Chemoinformatique/VQGAE.git
cd VQGAE/
```
If you haven't installed `conda-lock` package in your base enviroment, you can do it using the following command:
```bash
conda install --channel=conda-forge --name=base conda-lock
```

Then, you should create a new enviroment using `vqgae_gpu.yml` file:

```bash
conda env create --name vqgae_env --file vqgae_gpu.yml
```
Then, you should activate the created enviroment, download repository and install VQGAE:

```bash
conda activate vqgae_env
pip install .
```

### Manual installation

If drivers on your NVIDIA machine does not match with the ones used in enviroment, 
you can manually install all required packages.
(Currently, we used Pytorch for CUDA 11.8 while drivers were already version of 12.0 and it worked fine)

First, check your GPU driver version with `nvcc` or `nvidia-smi`.

In case you haven't installed cudatoolkit drivers, and it requires administrator permissions whic you might not have,
the only way to install is pytorch GPU version is with conda:

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

`pip install "pytorch-lightning>2.0" "adabelief-pytorch>=0.2.1"`

## Example usage

The tool work in command line mode. For the training you can simply run:

```bash
vqgae_train fit -c configs/vqgae_training.yaml
```

For the encoding you should run the following command:

```bash
vqgae_encode -c configs/vqgae_encode.yaml
```

And for the decoding you should run the following command:

```bash
vqgae_decode -c configs/vqgae_decode.yaml
```

Also, if you want to create an example of default config, simply run:
```bash
vqgae_default_config --task train
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

## Citation

Please make sure to cite this work if you find it useful:

```bibtex
@article{akhmetshin2023construction,
  title={Construction of order-independent molecular fragments space with vector quantised graph autoencoder},
  author={Akhmetshin, Timur and Lin, Albert and Madzhidov, Timur and Varnek, Alexandre},
  journal={ChemRxiv},
  publisher={Cambridge Open Engage},
  year={2023},
  note={This content is a preprint and has not been peer-reviewed.},
  doi={10.26434/chemrxiv-2023-5zmvw}
}
```

## Copyright

* [Tagir Akhmetshin ](tagirshin@gmail.com)
* [Arkadii Lin](arkadiyl18@gmail.com)
* [Timur Madzhidov](tmadzhidov@gmail.com)
* [Alexandre Varnek](varnek@unistra.fr)