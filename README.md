# noisy-hawkes-cumulants

Code for the paper *"Cumulants of Hawkes Processes are Robust to Observation Noise" (ICML 2021)*

# Installation instructions

To run the code, the following pre-requisites must be installed.

The `conda` command from [Anaconda](https://www.anaconda.com/) must be available. The code was tested with conda 4.8.5 on Mac OS 10.15.7.

First clone the repository
```
git clone https://github.com/trouleau/noisy-hawkes-cumulants.git
```

Then create a new environment.

```
# Create and activate environment
conda create -y -n env python=3.7
conda activate env

# Install main dependencies
pip install -r requirements.txt

# Install lib for algorithm Desync-MLE
git clone https://github.com/trouleau/desync-mhp desync_mhp/
cd desync-mhp/lib/model/_heavy/ && python setup.py build_ext --inplace && cd - 

# Install internal utility lib
cd lib && pip install -e . && cd -
```

Then install and start `jupyter-lab` to run the notebooks.

```
pip install jupyterlab
jupyter-lab notebooks
```

All the figures in the paper can be generate with the two documented notebooks in `notebooks/`. Further explainations are available inside each notebook.
