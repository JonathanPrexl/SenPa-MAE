FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

MAINTAINER Jonathan Prexl

RUN apt update && apt install -y \
  tmux \
  htop \
  build-essential \
  nano

RUN pip install --upgrade pip
RUN pip install fastprogress
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install tqdm
RUN pip install tensorboard
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install easydict
RUN pip install tensorboardX
RUN pip install torchinfo
RUN pip install argconfigparse
RUN pip install torchmetrics
RUN pip install seaborn
RUN pip install hydra-core
RUN pip install shapely
RUN pip install geopandas
RUN pip install xarray
RUN conda install rasterio
RUN pip install timm
RUN pip install einops
RUN pip install h5py

ARG HOST_UID
RUN useradd user -u $HOST_UID

RUN mkdir -p /home/user
RUN chown user /home/user

ENV workdir /home/user
WORKDIR ${workdir}

USER user

CMD bash
