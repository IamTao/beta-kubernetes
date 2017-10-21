FROM itamtao/base
MAINTAINER Tao Lin <itamtao@gmail.com>

# install Tensorflow GPU Version
RUN pip --no-cache-dir install --upgrade --ignore-installed  \
    https://pypi.python.org/packages/1e/be/b72fd4690c2a557ed4a253364df8a328860851008935b84dfe47615b6eb5/tensorflow_gpu-1.3.0rc0-cp35-cp35m-manylinux1_x86_64.whl#md5=e0d732610e060191f11c085b3e189cd8

# install pytorch
RUN git clone https://github.com/pytorch/pytorch.git /root/pytorch && cd /root/pytorch && \
    git submodule update --init && \
    pip install -r requirements.txt && \
    python setup.py install && rm -rf /root/pytorch
RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

############ user: user permission
USER $NB_USER
WORKDIR $HOME
