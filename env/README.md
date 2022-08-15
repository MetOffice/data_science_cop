# Setting up environments for The Data Science Community of Practice scipts/notebooks

The environments for code found in this repository, either in scripts or in notebooks, depending on the platform and type of work you are doing, are all primarily provided by conda environments.

These files provide pre-checked base environments that are a starting point for project environments into which other more project specific libraries can be added.

To create an environment from any of the *.yml* files, run the following command:
```bash
conda env create --file requirement.yml
```
Where you replace `requirements.yml` with the particular requirements file in the folder required for the files you want to run.
There several different environments as different packages are required for different activities or demonstrations, some of which are mutually incompatible or make for slow/difficult to resolve installations, so we have tried to keep each environment as simple as possible to improve performance, as well as fixing compatible major and minor package versions to help the evironment solve.

**The installation of environments with the yml files in conda can take a significant amount of memory, for this reason we present two methods to overcome this issue: Install the provided lock files which are system dependant, or Use mamba to install yml files.
<br><br>
If you are using a system supported by the existing lock files, this is the recommended method of install. Lock files specify versions and builds for all included libraries which completely avoid solving the evironment**

## Installing environments via provided lock files

To create an environment from any of the *.lock* files, run the following command which also requires a name for the environment to be specified:
```bash
conda create --name myenv --file requirements.lock
```

Lock file are generated on and for specific systems, and will fail to install on systems that are significantly different from the target. Each lock file includes a reference to the target system as follows:
*\*\_mo\_linux*.lock - Met Office internal linux "desktop" systems.<br>
A significantly different system in this case would be a system not using both the Linux-64 architecture and the Met Office conda channels. Generation of the lock files was performed in the Met Office VDI.<br><br>
Environment name suggestions can be found at the top of all .yml files

## Installing environments via mamba

To create an environment using mamba, run the following set of commands: <br>
First, using the desktop conda, create an environment that contains mamba
```bash
conda create --name mamba --channel conda-forge mamba
```
Now activate that environment
```bash
conda activate mamba
```
Now create your data science environment
```bash
mamba env create --file requirements.yml
```

## Use on Met Office systems

### Use on VDI

To use on VDI, just create the conda environment as above, activate and then either run the script required, or start a jupyter lab server. Eg: 
```bash 
jupyter lab 
```

### SPICE

To use a script on spice, you can simply activate the environment inside your SPICE script. To run the notebooks on spice, use of the Met Office Jupyter Hub installation is advised. Any conda environments you have installed in your homespace will automatically become available as kernels in JupyterHub.
