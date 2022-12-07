FROM python:3.9.15-slim

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

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
    vim \
    nano \
    tmux \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/* 

# Install requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --default-timeout=1000
RUN pip3 install torch -f https://download.pytorch.org/whl/cu117.html --default-timeout=1000 

# Copy code
RUN mkdir -p output/
RUN mkdir -p data/
COPY meta_kg/ meta_kg/
COPY run_maml.py .
COPY cli_maml.py .
COPY data/proofwriter/ data/proofwriter/
COPY data/clutrr_2_hop/ data/clutrr_2_hop/
COPY run_gpt2.sh .

ENTRYPOINT []
CMD ["/bin/bash"]