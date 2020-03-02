# TreFiDe - Trend Filter Denoising


TreFiDe is the software package accompanying the research publication
["Penalized matrix decomposition for denoising, compression, and improved demixing of 
functional imaging data"](https://doi.org/10.1101/334706). 

TreFiDe is an imporved appproach 
to compressing and denoising functional image data. The method is based on a spatially-localized 
penalized matrix decomposition (PMD) of the data to separate (low-dimensional) signal from 
(temporally-uncorrelated) noise. This approach can be applied in parallel on local spatial 
patches and is therefore highly scalable, does not impose non-negativity constraints or require 
stringent identifiability assumptions (leading to significantly more robust results compared to 
NMF), and estimates all parameters directly from the data, so no hand-tuning is required. We 
have applied the method to a wide range of functional imaging data (including one-photon, 
two-photon, three-photon, widefield, somatic, axonal, dendritic, calcium, and voltage imaging 
datasets): in all cases, we observe ~2-4x increases in SNR and compression rates of 20-300x 
with minimal visible loss of signal, with no adjustment of hyperparameters; this in turn 
facilitates the process of demixing the observed activity into contributions from individual 
neurons. We focus on two challenging applications: dendritic calcium imaging data and voltage 
imaging data in the context of optogenetic stimulation. In both cases, we show that our new 
approach leads to faster and much more robust extraction of activity from the video data.


## Install using conda

This installation method is supported and tested only on Ubuntu 18.04.

It is recommended to use [conda](https://www.anaconda.com/) to manage the 
dependencies for TreFiDe in it's own Python environment.
First, download and install [conda](https://www.anaconda.com/distribution/). Verify conda installation
by executing the following scripts. A list of base environment packages will be displayed.
```
conda list
```

<!-- pytorch only requires nvidia driver, doesn't require to install cuda. -->
Create a new environment for TreFiDe and install TreFiDe software and all of its dependencies. 
TreFiDe version 2.0 supports 4D (x,y,z,T) input image data. It is the latest development branch.
Version 2.0 changes and cleans up function call and is different form Version 1.* function calls.
If you are familiar with version 1.* function calls, please continue using version 1.* or start 
experimenting with new function call in version 2.0. 
To install the 2.0 version TreFiDe, use 

```
conda create -n trefide_2.0 python=3.6 trefide -c jw3132 -c conda-forge
```

To install TreFiDe version 1.2, which only supports 3D (x,y,T) input image data, use


```
conda create -n trefide_1.2 python=3.6 trefide=1.2 -c jw3132 -c conda-forge
```


Download `Demo` folder. Follow the notebook scripts in `Demo` folder to try out TreFiDe.
To learn about TreFiDe functions, please check [doc](http://htmlpreview.github.io/?https://github.com/ikinsella/trefide/blob/master/doc/trefide.html)
.

For user that doesn't have access to Ubuntu:18.04 but is familiar with [Docker](https://www.docker.com/),
please use Ubuntu docker image to access TreFiDe software. 


## Install from source

### Dependencies:
- A Linux OS, Ubuntu 16.04 recommended (Installation support for MacOS and Windows coming);
- Python3;
- Required Python Packages: numpy, scipy, cython;
- Recommended Python Packages (To run demos, generate plots, and render videos): matplotlib, jupyter, opencv3; 
- Intel MKL (see below for instructions);
- C compiler (installation currently requires ```icc```, support for ```gcc``` coming);

This package contains C++ source code with Cython wrappers which need to be built on your system. 
The easiest way to ensure all the required libraries are installed is to follow the instructions for installing & setting up [Intel MKL](https://software.intel.com/en-us/mkl) (which is a free product for both personal and commercial applications).
Additionally, you will need a C++ compiler. For ease of implementation, the current installation scripts are setup for the [Intel C Compiler](https://software.intel.com/en-us/c-compilers) ```icc``` (which is free for students and academics). Support to optionally use ```gcc``` (which is more commonly available by default) is coming.

### Installing:
Ensure that the neccessary dependencies are installed and that your the python environment you wish to install trefide into (we highly recommend using ```conda``` contained in the [Anaconda & Miniconda disctributions](https://www.anaconda.com/download/#linux) to manage your python environments) is active.
1. Clone the repository by navigating to the location you wish to install the package and executing```git clone git@github.com:ikinsella/trefide.git```. The absolute path to the location mentioned above will be refered to as ```/path/to/install/directory``` for the remainder of these instructions.
2. Add the location of the C++ libraries to your shared library path by appending the lines
```
export TREFIDE="/path/to/install/directory/trefide"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src/proxtv"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TREFIDE/src/glmgen/lib"
```
to your ```.bashrc``` file.
3. Compile the C++ source code by running 
```
cd /path/to/install/directory/trefide/src
make
```
4. Build the Cython wrappers and use pip to create an "editable" installation in your active python environment by running
```
cd /path/to/install/directory/trefide
LDSHARED="icc -shared" CC=icc CXX=icpc pip install -e /path/to/trefide
```
5. Execute PMD demo code using the sample data [here](https://drive.google.com/file/d/1v8E61-mKwyGNVPQFrLabsLsjA-l6D21E/view?usp=sharing) to ensure that the installation worked correctly.

### Rebuilding & Modification
If you modify or pull updates to any C++ &/or Cython code, the C++ &/or Cython code (respectively) will need to be rebuilt for changes to take effect. This can be done by running the following lines
- C++:
  ```
  cd /path/to/install/directory/trefide/src
  make clean
  make
  ```
- Cython:
  ```
  cd /path/to/install/directory/trefide
  LDSHARED="icpc -shared" CXX=icpc CC=icc python setup.py build_ext --inplace
  ``` 

### Uninstalling
The project can be uninstalled from an active python environment at any time by running ```pip uninstall trefide```. If you wish to remove the entire project (all of the files you cloned) from your system, you should also run ```rm -rf /path/to/install/directory/trefide```.

## References:
- [preprint](https://www.biorxiv.org/content/early/2018/06/03/334706.article-info)
- support [slack channel](https://join.slack.com/t/trefide/shared_invite/enQtMzc5NDM4MDk4OTgxLWE0NjNhZGE5N2VlMTcxNGEwODhkMmFlMjcyYmIzYTdkOGVkYThhNjdkMzEyZmM1NzIzYzc0NTZkYmVjMDY5ZTg)
