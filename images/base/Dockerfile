FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Tao Lin <itamtao@gmail.com>

# configure some environments.
RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

# install some necessary tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common
RUN apt-get install -y \
        inkscape \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libcupti-dev \
        pkg-config \
        libav-tools \
        libjpeg-dev zlib1g-dev \
        locales
RUN apt-get install -y \
        sudo \
        rsync \
        cmake \
        g++ \
        swig \
        vim \
        git \
        curl \
        wget \
        unzip \
        zsh \
        git \
        screen \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python3.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python \
        python-numpy \
        python-dev \
        python-pip \
        python-wheel \
        python-tk \
    && pip install --upgrade pip setuptools

# install some python packages.
RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        scipy \
        scikit-learn \
        sklearn \
        pandas \
        Pillow \
        h5py \
        autograd \
        powerlaw \
        seaborn \
        nltk \
        && \
    python -m ipykernel.kernelspec

# install good vim.
RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
ENV SHELL=/bin/bash \
    NB_USER=lin \
    NB_UID=144057 \
    NB_GROUP=MLO-unit \
    NB_GID=11169
ENV HOME=/home/$NB_USER

ADD base/fix-permissions /usr/local/bin/fix-permissions
RUN chmod +x /usr/local/bin/fix-permissions
RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm ${NB_USER} && \
    fix-permissions $HOME

# Set up our notebook config.
COPY base/jupyter_notebook_config.py /root/.jupyter/
COPY base/run_jupyter.sh /

# IPython
EXPOSE 8888
