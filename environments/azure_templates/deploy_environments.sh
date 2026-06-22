# this script demonstrates deploying different sorts of environments

# deploy from a conda file
az ml environment create -f environments/azure_templates/envs/env_keras_conda_aml.yaml -w mlw-dscoptrainingenv-uksouth-01 -g rg-dscop-dscoptrainingenv

# deploy from a dockerfile
az ml environment create -f environments/azure_templates/envs/keras/env_keras_aml.yaml -w mlw-dscoptrainingenv-uksouth-01 -g rg-dscop-dscoptrainingenv
