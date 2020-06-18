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
exactly as it was for the paper. Repo `dps` and `auto_yolo` should also be
on branch `aaai_2020` (the install script `install.sh` installs these repositories
as well, and will attempt to checkout branches in those repos with the same
name as the current branch for this repo).

##  Installation
1. [Install tensorflow](https://www.tensorflow.org/install/) with [GPU support](https://www.tensorflow.org/install/gpu).
   SILOT was developed with tensorflow 1.13.2 and CUDA 10.0; no guarantees that it will work
   with other versions. In particular, versions of tensorflow >= 1.14 introduce changes to the process of building custom
   tensorflow ops (of which this repo relies on 2, both in the `auto_yolo` dependency, see below) which this repo does not yet take into account.

   Example:
   ```
   pip install tensorflow_gpu==1.13.2
   ```

2. `sh install.sh`

   This downloads a number of dependency repos and installs them and *their* dependencies. These repos are created inside
   the SILOT repo, but are not *tracked* by the SILOT repo (achieved by .gitignore).

   1. `dps`: custom framework for managing datasets and the training loop.
   2. `auto_yolo`: tensorflow implementation of SPAIR, which is used roughly as a sub-module within SILOT.
   3. `sqair`: tensorflow implementation of Sequential Attend, Infer, Repeat, a predecessor of SILOT.

3. Install a version of `tensorflow_probability` that matches your version of tensorflow (0.6 works for tensorflow 1.13, increment by 0.1 for each 0.1 increment of tf version).

   Example:
   ```
   pip install tensorflow_probability==0.6
   ```

## Training SILOT

### Moving MNIST
```
cd silot
python run.py moving_mnist silot
```
When this is run for the first time, `dps` will download EMNIST data required
for building the Moving MNIST dataset. It will also create a dataset of 60,000
examples, which can take a while. To use fewer examples, one can do:
```
python run.py moving_mnist silot --n-train=1000
```

### Moving Shapes
```
cd silot
python run.py shapes shapes_silot
```
This will create a dataset of 60,000 examples, which can take a while. To use fewer examples, one can do:
```
python run.py shapes shapes_silot --n-train=1000
```

### Atari
We first need to download the atari data and unzip it before we can run silot on it.
```
sh download_atari_data.sh
```
This downloads atari data (in the form of a github repo called `atari_rollouts`)
into the current working directory; silot assumes this is done inside the `silot/silot` directory
(i.e. the same directory where the `run.py` script lives). If you want to download the data
into another location, you'll need to tell silot where the data lives, which
can be achieved by editing the `atari_data_dir` variable in `run.py`.

We can then run silot on any of `asteroids, carnival, space_invaders, wizard_of_wor`.
E.g.:
```
python run.py space_invaders atari_train_silot
```

## Viewing Results

SILOT uses a custom framework called `dps` to manage datasets and run the training loop.
By default, dps will store data (cached datasets and experiment results) in
a directory created at path "./dps_data" (i.e. will create a directory called
`dps_data` inside the current working directory when you run the experiment).
To change this location, find the `scratch_dir` entry in silot/run.py, and edit it to point at your desired location.
Hereafter we will refer to this location as `dps_data`.

Experimental results are stored in `dps_data/local_experiments`. Within
`dps_data/local_experiments`, a directory will be created for each environment
(e.g. mnist, shapes, atari) as the relevant experiments are run.
Within those directories, a new directory will be created for each new experiment.
Each experiment directory contains a number of files, including a log of stdout and
stderr for the experiment, the config that the experiment was run under,
weights saved at various points, visualizations, etc.

Looking at the visualizations generated as the model trains is probably the most useful
way to diagnose model performance. The frequency with which visualizations are
created is controlled by `render_step` in `silot/run.py`. These are stored in
the `plots` sub-directory of each experiment.

A number of diagnostic values are recorded throughout training and may be viewed using `tensorboard`.
These values are stored inside the `summaries` sub-directory of each experiment
directory. By default `dps` spins up tensorboard for each new experiment;
results may be accessed by navigating to `localhost:6006` in your browser. The
frequency with which dps writes tensorboard-viewable summaries can be controlled
by `eval_step` in `silot/run.py`.
