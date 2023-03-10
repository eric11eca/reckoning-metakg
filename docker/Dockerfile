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

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

WORKDIR /

# Find your username by running `id -un` on HaaS and fill it here
ENV USER_NAME=<your-username>
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} && \
    chown ${USER_ID} -R /home/${USER_NAME} && \
    # Change the password to make root > user
    echo -e "${USER_NAME}\n${USER_NAME}" | passwd ${USER_NAME}

# Set some basic ENV vars for readability
# NOTE: IF YOU ARE NOT USING THE NFS FEEL FREE TO MAKE THE FOLLOWING HOME VAR JUST ROOT
ENV HOME=/home/${USER_NAME}
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV REPO_DIR=${HOME}/<repo-name>

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

# WORKDIR instruction sets the directory the following instructions should be run from
WORKDIR ${HOME}

# Install conda (optional)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN ${CONDA} create --name myenv python=3.9

# Git configuration (optional, you can also use a repo saved in the NFS)
# RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
#     git config --global user.name ${GITHUB_NAME}
# RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
#     git config --global user.email ${GITHUB_EMAIL}
# # RUN git config --global pull.rebase false

# # Clone your github repo (optional)
# RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
#     git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/${GITHUB_USERNAME}/<repo-name>.git
# # Make this repo your WORKDIR
# WORKDIR ${REPO_DIR}

# Setup github repo dependencies
# Note that, you cannot activate an environment with the RUN command, as each RUN command is like a new session.
# Thus we instead call conda at the beginning of the pip or python command the way shown here:

WORKDIR /nlp
RUN ${CONDA} run -n myenv pip install -r <repo-name>/requirements.txt
RUN ${CONDA} run -n myenv pip install -e .
RUN ${CONDA} run -n myenv python -c "import nltk;nltk.download('punkt');nltk.download('wordnet');nltk.download('omw-1.4')"

# # Login to wandb
# RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
#     ${CONDA} run -n myenv wandb login ${WANDB_API_KEY}

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd
# RUN echo 'root:root' | chpasswd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

# Install the Run:AI Python library and its dependencies
RUN ${CONDA} run -n myenv pip install runai prometheus_client==0.7.1

# Prepare the NFS mount folder
RUN mkdir /mnt/nlpdata1
RUN mkdir /mnt/scratch

# NOTE: IF YOU ARE NOT USING THE NFS FEEL FREE TO REMOVE THE FOLLOWING 2 INSTRUCTIONS
# Changing the ownership of the /home/USER folder, so that the files created by root can be accesible (e.g. git cloned repo)
# Otherwise by default, they are owned by root
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    chown ${USER_ID} -R /home/${USER_NAME} 

# Switch to user instead of root for NFS + home directory access
USER ${USER_NAME}

RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["/usr/sbin/sshd", "-D"]
# ENTRYPOINT ["./entrypoint.sh"]
