###############################################################################
# Nvidia base docker image where all drivers are installed
###############################################################################
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ENV CUDA_VISIBLE_DEVICES=0

###############################################################################
# Small fix for the internal time of the container. Logging has issues otherwise
###############################################################################
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

##############################################################################
# Arguments for the container setup
##############################################################################
ARG WANDB_KEY

RUN [ ! -z "${WANDB_KEY}" ]

##############################################################################
# Installing system libraries
##############################################################################
RUN apt-get -y update
RUN apt-get -y install \
        tzdata git wget cmake ninja-build make zip curl unzip python3-dev python3-pip python3-venv pkg-config build-essential \
        libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm \
        libncursesw5-dev xz-utils tk-dev libxml2-dev \
        libxmlsec1-dev libffi-dev liblzma-dev \
        autoconf libtool flex bison \
        rsync libedit-dev htop ffmpeg git &&\
    apt-get clean

##############################################################################
# Installing Mesa for Mujoco headless rendering
##############################################################################
RUN apt-get -y install \
         libdrm-dev \
         libwayland-dev \
         libwayland-egl-backend-dev \
         libxfixes-dev \
         libxcb-glx0-dev \
         libxcb-shm0-dev \
         libx11-xcb-dev \
         libxcb-dri2-0-dev \
         libxcb-dri3-dev \
         libxcb-present-dev \
         libxshmfence-dev \
         libxxf86vm-dev \
         libxrandr-dev
RUN apt-get clean

RUN pip3 install meson packaging Mako

RUN mkdir mesa &&\
    cd mesa &&\
    wget https://mesa.freedesktop.org/archive/mesa-24.1.2.tar.xz &&\
    tar xvf mesa-24.1.2.tar.xz &&\
    cd mesa-24.1.2 &&\
    meson setup builddir -Dosmesa=true -Dgallium-drivers=swrast -Dvulkan-drivers=[] -Dprefix=$PWD/builddir/install &&\
    meson install -C builddir

WORKDIR /

#######################################################################
# Setup virtual environment
#######################################################################
RUN python3 -m venv venv &&\
    echo "source /venv/bin/activate" >> ~/.bashrc

COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN /venv/bin/pip3 install -r requirements.txt &&\
    rm requirements.txt &&\
    rm setup.py

#######################################################################
# System variables
#######################################################################
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mesa/mesa-24.1.2/builddir/install/lib/x86_64-linux-gnu

#######################################################################
# WandB setup
########################################################################
RUN pip3 install wandb &&\
    python3 -m wandb login ${WANDB_KEY}


#######################################################################
# Container entrypoint configuration
########################################################################
COPY /scripts/docker_entrypoint.sh /scripts/docker_entrypoint.sh
RUN ["chmod", "+x", "/scripts/docker_entrypoint.sh"]
ENTRYPOINT ["/scripts/docker_entrypoint.sh"]