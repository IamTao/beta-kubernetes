FROM itamtao/base
MAINTAINER Tao Lin <itamtao@gmail.com>

USER $NB_USER
WORKDIR $HOME

# install openMPI
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH


# install conda
ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     sh miniconda.sh -b -p $HOME/conda && \
     rm ~/miniconda.sh
RUN  $HOME/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl setuptools cmake cffi && \
     $HOME/conda/bin/conda clean -ya
ENV PATH $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
RUN conda install --name pytorch-py$PYTHON_VERSION -c soumith magma-cuda80

RUN git clone --recursive https://github.com/pytorch/pytorch
RUN cd pytorch && \
    git submodule update --init && \
    TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .
