<div align="center">
<img align="middle" src="etc/concept-1.png" width="700" alt="logo"/>
<h2>Neural Canonical Transformation </h2>
</div>

PyTorch implement of the paper: Neural Canonical Transformation with Symplectic Flows. 

A sympletic normalizing flow which captures collective modes in the latent space. It identifies slow collective variables and performs conceptual compression. 



## Usage

### 1. Phase Space Density Estimation

#### 1.1 Molecular Dynamics Trajectory data

To train a neuralCT for molecular dynamics data, use `mdmain.py`
```bash
python ./mdmain.py -cuda 6 -batch 200 -epoch 500 -fixy 2.3222 -dataset ./database/alanine-dipeptide-3x250ns-heavy-atom-positions.npz
```

**Options**

- **-epochs**: Number of epoches to train;
- **-batch**: Batch size of the training;
- **-cuda**: Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU;
- **-hdim**: Hidden dimension of mlps;
- **-numFlow**: Number of flows layers;
- **-nlayers**: Number of mlps layers in the rnvp;
- **-nmlp**: Number of dense layers in each mlp;
- **-smile**: SMILE expression of this molecular;
- **-scaling**: Scaling factor of npz data, default is 10(for nm to ångströms);
- **-fixx/y/z**: Offset of x/y/z axis of data from npz file;
- **-dataset**: Path to the training data.

To see detailed options, run`python main.py -h`.

**Notebook**

[Alanine Dipeptide](3_AlanineDipeptide.ipynb)

#### 1.2  Image dataset

To train a neuralCT for machine learning dataset, use `mlmain.py`. 
```bash
python ./mlmain.py -epochs 5000 -batch 200 -cuda 1 -hdim 256 -nmlp 3 -nlayers 16 -dataset ./database/mnist.npz
```

**Options**

- **-epochs**: Number of epoches to train;
- **-batch**: Batch size of the training;
- **-cuda**: Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU;
- **-hdim**: Hidden dimension of mlps;
- **-numFlow**: Number of flows layers;
- **-nlayers**: Number of mlps layers in the rnvp;
- **-nmlp**: Number of dense layers in each mlp;
- **-n**: Number of dimensions of the training data;
- **-dataset**: Path to the training data.

To see detailed options, run`python main.py -h`.

**Notebook**

[MNIST concept compression](4_MNIST.ipynb)




### 2. Variational Free Energy

To train a neuralCT via the variational approach, use `variationalMain.py`. To train on our provided target distributios, you can specify this using **-source** option. To train on a different taget distribution, you will have to code your own target, examples can be found in `source` folder.
```bash
python ./variationMain.py -epochs 5000 -batch 200 -cuda 1 -hdim 256 -nmlp 3 -nlayers 16 -source 0
```

**Options**

- **-epochs**: Number of epoches to train;
- **-batch**: Batch size of the training;
- **-cuda**: Which device to use with -1 standing for CPU, number bigger than -1 is N.O. of GPU;
- **-hdim**: Hidden dimension of mlps;
- **-numFlow**: Number of flows layers;
- **-nlayers**: Number of mlps layers in the rnvp;
- **-nmlp**: Number of dense layers in each mlp;
- **-source**: Using which source, 0 for Ring2d, 1 for HarmonicChain.

To see detailed options, run `python main.py -h`.

**Notebooks**

1. [Ring2D distribution](1_Ringworld.ipynb)
2. [Harmonic Chain](2_HarmonicChain.ipynb)

## Citation

````latex
@article{neuralCT,
  Author = {Shuo-Hui Li, Chen-Xiao Dong, Linfeng Zhang, and Lei Wang},
  Title = {Neural Canonical Transformation with Symplectic Flows},
  Year = {2019},
  Eprint = {arXiv:XXXX.XXXXX},
}
````

## Contact

For questions and suggestions, contact Shuo-Hui Li at [contact_lish@iphy.ac.cn](mailto:contact_lish@iphy.ac.cn).
