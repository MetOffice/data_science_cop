FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

RUN conda create -n lightning python=3.11 -y && \
    conda run -n lightning pip install --no-cache-dir \
      torch \
      torchvision \
      torchaudio \
      lightning \
      pytorch-lightning \
      pandas \
      numpy \
      scikit-learn \
      azureml-mlflow && \
    conda clean -afy

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/lightning
ENV PATH=/opt/miniconda/envs/lightning/bin:$PATH