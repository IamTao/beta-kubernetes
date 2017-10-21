FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
MAINTAINER Tao Lin <itamtao@gmail.com>

RUN "sh" "-c" "echo nameserver 8.8.8.8 >> /etc/resolv.conf"

############ user: root permission
USER root

# Pick up some dependencies
RUN apt-get update && apt-get -yq dist-upgrade \
        && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        software-properties-common \
        inkscape \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        libav-tools \
        sudo \
        cmake \
        rsync \
        git \
        curl \
        wget \
        unzip \
        zsh \
        libjpeg-dev libfreetype6-dev zlib1g-dev \
        octave pandoc \
        vim emacs \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -yq dist-upgrade \
 && apt-get install -yq --no-install-recommends \
    bzip2 \
    locales \
    fonts-liberation \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure vim
RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# Install Tini
RUN wget --quiet https://github.com/krallin/tini/releases/download/v0.10.0/tini && \
    echo "1361527f39190a7338a0b434bd8c88ff7233ce7b9a4876f3315c22fce7eca1b0 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# Configure environment
ENV MINICONDA_VERSION 4.3.21
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=mlo \
    NB_UID=144057 \
    NB_GROUP=lin \
    NB_GID=11169 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

# Create user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
ADD ml/fix-permissions /usr/local/bin/fix-permissions
RUN chmod +x /usr/local/bin/fix-permissions
RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm ${NB_USER} && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

############ user: user permission
# USER $NB_USER

# Setup work directory for backward-compatibility
RUN mkdir $HOME/notebooks && fix-permissions $HOME

# Install conda as mlo and check the md5 sum provided on the download site
RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "c1c15d3baba15bf50293ae963abef853 *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda update --all --quiet --yes && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR

# Install Jupyter Notebook and Hub
RUN conda install --quiet --yes \
    'notebook=5.1.*' \
    'jupyterhub=0.7.*' \
    'jupyterlab=0.27.*' \
    && conda clean -tipsy && \
    fix-permissions $CONDA_DIR

# Install Python 3 packages
# Remove pyqt and qt pulled in for matplotlib since we're only ever going to
# use notebook-friendly backends in these images
RUN conda install --quiet --yes \
    'nomkl' \
    'ipywidgets=7.0*' \
    'pandas=0.19*' \
    'numexpr=2.6*' \
    'matplotlib=2.0*' \
    'scipy=0.19*' \
    'seaborn=0.7*' \
    'scikit-learn=0.18*' \
    'scikit-image=0.12*' \
    'sympy=1.0*' \
    'cython=0.25*' \
    'patsy=0.4*' \
    'statsmodels=0.8*' \
    'cloudpickle=0.2*' \
    'dill=0.2*' \
    'numba=0.31*' \
    'bokeh=0.12*' \
    'sqlalchemy=1.1*' \
    'hdf5=1.8.17' \
    'h5py=2.6*' \
    'vincent=0.4.*' \
    'beautifulsoup4=4.5.*' \
    'protobuf=3.*' \
    'xlrd'  && \
    conda remove --quiet --yes --force qt pyqt && \
    conda clean -tipsy && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    fix-permissions $CONDA_DIR

# Install facets which does not have a pip or conda package at the moment
RUN cd /tmp && \
    git clone https://github.com/PAIR-code/facets.git && \
    cd facets && \
    jupyter nbextension install facets-dist/ --sys-prefix && \
    rm -rf facets && \
    fix-permissions $CONDA_DIR

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME $HOME/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions $HOME

# install tensorflow
RUN conda install --quiet --yes \
    'tensorflow=1.3*' \
    'keras=2.0*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR

# install torch
ENV CMAKE_PREFIX_PATH "$(dirname $(which conda))/../"  # [anaconda root directory]
RUN conda install --quiet --yes numpy pyyaml mkl setuptools cmake gcc cffi
RUN conda install --quiet --yes -c soumith magma-cuda80  # or magma-cuda75 if CUDA 7.5
RUN conda install --quiet --yes pytorch torchvision

############ user: root permission
# USER root
WORKDIR $HOME/notebooks

# Add local files as late as possible to avoid cache busting
COPY ml/start.sh /usr/local/bin/
COPY ml/start-notebook.sh /usr/local/bin/
COPY ml/start-singleuser.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*
COPY ml/jupyter_notebook_config.py /etc/jupyter/
RUN fix-permissions /etc/jupyter/

# Configure container startup
# ENTRYPOINT ["tini", "--"]
# CMD ["start-notebook.sh"]

############ user: user permission
USER $NB_USER
