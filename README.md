# Spatially Invariant, Label-free Object Tracking (SILOT)

SILOT is Variational Autoencoder for videos that models videos as collections
of moving objects. It is able to scale to many 10s of objects (and likely even
beyond that), and achieves this through extensive use of spatially invariant
representations and computations.
Examples of what SILOT can achieve can be found [here](https://sites.google.com/view/silot).

This repo contains code for running experiments found in the following paper:

Exploiting Spatial Invariance for Scalable Unsupervised Object Tracking.  
Eric Crawford and Joelle Pineau.  
*AAAI (2020).*
```
@inproceedings{crawford2020exploiting,  
  title={Exploiting Spatial Invariance for Scalable Unsupervised Object Tracking},  
  author={Crawford, Eric and Pineau, Joelle},  
  booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},  
  year={2020}  
}
```

This repo, and some of those it depends on (`dps` and `auto_yolo`), will likely undergo further
development in the future. We will always attempt to keep the experiments from
the above paper runnable. However, failing that, one can checkout the branch
`aaai_2020` before running the installation procedure below to get the code
exactly as it was for the paper. Repositories dps and auto_yolo should also be
on branch `aaai_2020` (the install script `install.sh` installs these repositories
as well, and will attempt to checkout branches in those repos with the same
name as the current branch for this repo).

###  Installation:
1. [Install tensorflow](https://www.tensorflow.org/install/) with [GPU support](https://www.tensorflow.org/install/gpu).
   SILOT was developed with tensorflow 1.13.2 and CUDA 10.0; no guarantees that it will work
   with other versions, though it probably will.

2. `sh install.sh`

3. Install a version of `tensorflow_probability` that matches your version of tensorflow (0.6 works for tensorflow 1.13, increment by 0.1 for each 0.1 increment of tf version).

### Start Training a SILOT model on Moving MNIST:
```
cd silot
python run.py moving_mnist silot
```

### SILOT on Atari
```
cd silot
python run.py space_invaders silot
```

Full experiments in silot/experiments.
