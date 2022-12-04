FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu18.04

# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# ENV PATH /usr/local/nvidia/bin/:$PATH
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /nlp

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/* 

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --default-timeout=1000
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116 --default-timeout=1000 

# Copy code
RUN mkdir -p output/
RUN mkdir -p data/
COPY meta_kg/ meta_kg/
COPY data/clutrr_simple/ data/clutrr_simple/
COPY run_gpt2.sh .

ENTRYPOINT []
CMD ["/bin/bash"]