#!/usr/bin/env bash

# This script should be run from the root of the dscop repository
export DSCOP_REPO_ROOT=$(pwd)
export DSCOP_DATA_DIR=/data/users/dscop/
#
# specify details of the dscop workspace
export RESOURCE_GROUP=rg-dscop-readiness-dscop-readiness-2h0pjp1nbi
export WORKSPACE_NAME=mlw-2h0pjp1nbi-uksouth-01

export STORAGE_ACCOUNT=sad5tm4cgkoum5401 
export CONTAINER_NAME=azureml-blobstore-3254a7ba-a602-44a8-9a2a-1888f1462c67

export CONTAINER_URL=https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}

# specify details of the source data to be copied
export XBT_DATA_DIR=${DSCOP_DATA_ROOT}/ml_tutorial/xbt/
export XBT_RANGE=(1966 2015)

export CLIMATE_ZONES_DATA_PATH=${DSCOP_DATA_DIR}/ml_tutorial/climate_zones/ml_ready/

export WEATHERBENCH_DIR=${DSCOP_DATA_DIR}/weatherbench/5.625deg
export WEATHERBENCH_VARS=(temperature specific_humidity geopotential u_component_of_wind v_component_of_wind total_precipitation total_cloud_cover constants 2m_temperature potential vorticity)


#copy data to datastore
#=========================
# xbt
for year in $(seq ${XBT_RANGE[0]} ${XBT_RANGE[1]});
    do azcopy copy ${XBT_DATA_DIR}/xbt_${year}.csv ${CONTAINER_URL}/xbt/ 
done;

# era5 - weatherbench 5 degree
for var_name in ${WEATHERBENCH_VARS[@]};
    do echo ${var_name}
    azcopy copy --recursive="true" "${WEATHERBENCH_DIR}/${var_name}/*.nc" "${CONTAINER_URL}/weatherbench/5.625deg/${var_name}/" 
done

#climate zones
export CLIMATE_ZONES_AML_URI=${CONTAINER_URL}/climate_zones/
azcopy copy --recursive ${CLIMATE_ZONES_DATA_PATH} ${CLIMATE_ZONES_AML_URI}

# cmip6 - hadgem3-LL

# haduk grid

# midas open

# mlcast uk radar
#=================================================
# create data assets

python environments/azure_templates/create_workspace_spec.py dataset --name climate_zones_1_0 --path ${CLIMATE_ZONES_AML_URI} --type uri_folder --output dataset_cz_1_0.yaml
az ml data create --file dataset_cz_1_0.yaml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

# ======================
# create environments

# tensorflow keras

# pytorch

# pyearthtools

# anemoi

# earth2studio

