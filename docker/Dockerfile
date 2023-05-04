FROM python:3.10.11-slim

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

SHELL ["/bin/bash", "-cu"]

# ENV GROUP_ID=1000

# WORKDIR /

# ENV USER_NAME=<your-username>
# RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
#     useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} && \
#     chown ${USER_ID} -R /home/${USER_NAME} && \
#     echo -e "${USER_NAME}\n${USER_NAME}" | passwd ${USER_NAME}

# ENV HOME=/home/${USER_NAME}
# ENV CONDA_PREFIX=${HOME}/.conda
# ENV CONDA=${CONDA_PREFIX}/condabin/conda
# ENV REPO_DIR=${HOME}/<repo-name>


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

# WORKDIR ${HOME}

# # Install conda (optional)
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# RUN bash miniconda.sh -b -p ${CONDA_PREFIX}
# RUN ${CONDA} config --set auto_activate_base false
# RUN ${CONDA} init bash
# RUN ${CONDA} create --name myenv python=3.10.11


# WORKDIR ${REPO_DIR}

WORKDIR /nlp
# RUN ${CONDA} run -n myenv pip install -r <repo-name>/requirements.txt

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --default-timeout=1000
RUN git clone https://github.com/facebookresearch/higher.git
RUN cd higher && pip3 install .


# RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
#     mkdir -p /var/run/sshd
# # RUN echo 'root:root' | chpasswd
# RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
# ENV NOTVISIBLE="in users profile"
# RUN echo "export VISIBLE=now" >> /etc/profile
# EXPOSE 22

# RUN ${CONDA} run -n myenv pip install runai prometheus_client==0.7.1

# Prepare the NFS mount folder
# RUN mkdir /mnt/nlpdata1
# RUN mkdir /mnt/scratch

# RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
#     chown ${USER_ID} -R /home/${USER_NAME} 
# USER ${USER_NAME}

ARG DUMMY=unknown
RUN DUMMY=${DUMMY}
RUN mkdir -p /nlp
RUN mkdir -p /nlp/output/
RUN mkdir -p /nlp/data/
COPY meta_kg/ /nlp/meta_kg/
COPY scripts/ /nlp/scripts/
COPY config /nlp/config
COPY run_maml.py /nlp/
COPY run_cli.py /nlp/

ENV WANDB_API_KEY 9edee5b624841e10c88fcf161d8dc54d4efbee29

CMD ["/bin/bash"]