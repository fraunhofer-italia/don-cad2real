FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ARG USER=don
ARG WS_DIR=/don_training

RUN groupadd ${USER} && \
    useradd -m -g ${USER} -s /bin/bash ${USER}

RUN mkdir -p ${WS_DIR} && \
    chown -R ${USER}:${USER} ${WS_DIR}

# Installation of python pip git and wget and additional
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    nano \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    libegl1 \
    cmake \
    python-is-python3
    
# Install additional packages for GUI applications
RUN apt-get update && apt-get install -y \
    libx11-xcb1 \
    libxcb1 \
    libxcb-glx0 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-sync1 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    x11-xserver-utils \
    xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix python symlink
RUN pip3 install --upgrade pip

# Move into working directory
WORKDIR ${WS_DIR}

# Install the 'dense_object_nets' package inside the container
COPY "dense_objects_nets" "./dense_objects_nets"
COPY "requirements.txt" "./requirements.txt"
COPY "pyproject.toml" "./pyproject.toml"
RUN pip install -e .

# Preload `swrast` and `iris` drivers
RUN echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6" >> ~/.bashrc

# Switch to the created user
USER ${USER}

CMD ["bash"]