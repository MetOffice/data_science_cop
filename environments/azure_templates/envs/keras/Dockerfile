FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN conda create -n keras-v3 python=3.11 -y && \
    conda run -n keras-v3 pip install --no-cache-dir \
      keras>=3 \
      tensorflow \
      pandas \
      numpy \
      scikit-learn \
      azureml-mlflow && \
    conda clean -afy

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/keras-v3
ENV PATH=/opt/miniconda/envs/keras-v3/bin:$PATH