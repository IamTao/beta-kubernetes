FROM itamtao/base
MAINTAINER Tao Lin <itamtao@gmail.com>

# install Tensorflow GPU Version
ENV TENSORFLOW_VERSION 1.3.0
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

# install pytorch
RUN pip --no-cache-dir install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
RUN pip --no-cache-dir install torchvision

############ user: user permission
USER $NB_USER
WORKDIR $HOME
