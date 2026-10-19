FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.8-cuda12.6:1

COPY requirements.txt /tmp/esci-training-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/esci-training-requirements.txt

