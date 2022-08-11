# Setting up environments for The Data Science Community of Practice scipts/notebooks

The environments for code found in this repository, either in scripts or in notebooks, depending on the platform and type of work you are doing, are all primarily provided by conda environments.

To create an environment from any of the *.yml* files, run the following command
```bash
conda env create --file requirement.yml
```
where you replace `requirements.yml` with the particular requirements file in the folder required for the files you want to run. There several different environments as different packages are required for different activities or demonstrations, some of which are mutually incompatible or make for slow/difficult to resolve installations, so we have tried to keep each environment as simple as possible to improve performance, as well as fixing compatible major and minor package versions to help environment solving.

**The installation of environments with the yml files can take a significant amount of memory, for this reason we present two methods to overcome this issue: Use mamba to install yml files, or Install the provided lock files which are system dependant.**

#### Installing environments via mamba

To create an environment using mamba, run the following set of commands: </br>
First, using the desktop conda, create an environment that contains mamba:
```bash
conda create --name mamba --channel conda-forge mamba
```
Now activate that environment:
```bash
conda activate mamba
```
Now create your data science environment:
```bash
mamba env create --file requirements.yml
```

#### Installing environments via provided lock files

To create an environment from any of the *.lock* files, run the following command which also requires a name for the environment to be specified
```bash
conda create --name myenv --file requirements.lock
```
environment name suggestions can be found at the top of all .yml files

## Use on Met Office systems

### Use on VDI

To use on VDI, just create the conda environment as above, activate and then either run the script required, or start a jupyter lab server.

### SPICE

To use a script on spice, you can simply activate the environment inside your SPICE script. To run the notebooks on spice, use of the Met Office Jupyter Hub installation is advised. Any conda environments you have installed in your homespace wll automatically become available as kernels in JupyterHub.
