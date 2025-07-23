# Setting up environments for The Data Science Community of Practice scripts/notebooks

The environments for code found in this repository, either in scripts or in notebooks, depending on the platform and type of work you are doing, are all primarily provided by conda environments.

These files provide pre-checked base environments that are a starting point for project environments into which other more project specific libraries can be added.

There are a variety of ways to make use of these environments
* In the Met Office - use pre-installed environments on the Community of Practice shared linux acount.
  


## Usage 

### Pre-installed DS CoP account (Met Office internal)

These have been installed on the Data Science Community of Practice shared account. These can be run following instructions on this [internal webpage](https://wwwspice/~dscop/environments.html)


### Clean install  
To create an environment from any of the *.yml* files, run the following command:
```bash
conda env create --file requirement.yml
```
Where you replace `requirements.yml` with the particular requirements file in the folder required for the files you want to run.
There several different environments as different packages are required for different activities or demonstrations, some of which are mutually incompatible or make for slow/difficult to resolve installations, so we have tried to keep each environment as simple as possible to improve performance, as well as fixing compatible major and minor package versions to help the evironment solve.

### Faster install from lock file

**The installation of environments with the yml files in conda can take a significant amount of memory, for this reason we present two methods to overcome this issue: Install the provided lock files which are system dependant, or Use mamba to install yml files.
<br><br>
If you are using a system supported by the existing lock files, this is the recommended method of install. Lock files specify versions and builds for all included libraries which completely avoid solving the evironment**

To create an environment from any of the *.lock* files, run the following command which also requires a name for the environment to be specified:
```bash
conda create --name myenv --file requirements_pytorch_molinux.lock
```

Lock file are generated on and for specific systems, and will fail to install on systems that are significantly different from the target. Each lock file includes a reference to the target system as follows:
*\*\_mo\_linux*.lock - Met Office internal linux "desktop" systems.<br>
A significantly different system in this case would be a system not using both the Linux-64 architecture and the Met Office conda channels. Generation of the lock files was performed in the Met Office VDI.<br><br>
Environment name suggestions can be found at the top of all .yml files


## Details of Use on Met Office systems

### Use on VDI

To use on VDI, just load or create the conda environment as above, activate and then either run the script required, or start a jupyter lab server. Eg: 
```bash 
jupyter lab 
```

### SPICE

To use a script on spice, you can simply activate the environment inside your SPICE script. To run the notebooks on spice, use of the Met Office Jupyter Hub installation is advised. Any conda environments you have installed in your homespace will automatically become available as kernels in JupyterHub.


## Admin

The script `install_central.sh` is used for easily installing all the environments in a single account with one command. The average user shouldn't need to use this, it is intended for the DS CoP admin team.