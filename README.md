# Data Science Community of Practice
This is the repository for example and tutorial material created through the Met Office Data Science Community of Practice.


## Running the code

Environment definitions have been provided for running different bits of code in the repoistory. These can be found in the [env](env) folder, which also contains instrcutions on setting up environments. These are intended ofr use in any enviornment where conda is available, such as 
* local desktop
* Jupyter Hub installation (e.g. [AWS Sagemaker](https://aws.amazon.com/sagemaker/), [AzureML](https://azure.microsoft.com/en-gb/services/machine-learning/#product-overview), [BinderHub](https://binderhub.readthedocs.io/en/latest/))
* cloud compute environment (e.g. [AWS](https://aws.amazon.com/), [Azure](https://azure.microsoft.com/en-gb/), [GCP](https://cloud.google.com/))

### Met Office

For users inside the Met Office, you can also use the default scitools environment for some of the notebooks. 

To run a local jupyter lab instance, the steps are:
* In a terminal, navigate to the repository `cd data_science_cop/`
* Load the `experimental-current` scitools environment `module load scitools`
* Run Jupyter Lab `jupyter lab`
* Navigate to the relevant notebook and run it.

You can alson run this through the  Jupyter Hub installation. Instructions on using JupyterHub with custoim conda enviornments can be found in the [env folder](env).
