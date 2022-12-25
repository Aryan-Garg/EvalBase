# For more information, please refer to https://aka.ms/vscode-docker-python

# FROM nvidia/cuda:11.1.1-base-ubuntu18.04
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /workspace

# Update the box (refresh apt-get)
RUN apt-get update -y

# Install apt-get requirements
RUN apt-get install -y tmux htop
RUN apt-get install -y wget curl iputils-ping
RUN apt-get install -y zip unzip nano
RUN apt-get install -y git build-essential cmake pkg-config

# Install Python
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN update-alternatives --install /usr/bin/python python $(which python3) 1

# OpenCV
RUN apt-get install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev

# Install Pillow
RUN apt-get install -y libjpeg8-dev zlib1g-dev 
RUN pip3 install --upgrade pip Pillow

# Other requirements
COPY ./.docker/requirements_aptGet.txt .
RUN xargs apt-get install -y < ./requirements_aptGet.txt

# VS Code Server
# Note, not necessary, but significant time saver
RUN wget "https://update.code.visualstudio.com/latest/server-linux-x64/stable" -O /tmp/vscode-server-linux-x64.tar.gz \  
    && mkdir /tmp/vscode-server \  
    && tar --no-same-owner -zxvf /tmp/vscode-server-linux-x64.tar.gz -C /tmp/vscode-server --strip-components=1 \  
    && commit_id=$(cat /tmp/vscode-server/product.json | grep '"commit":' | sed -E 's/.*"([^"]+)".*/\1/') \  
    && mkdir -p ~/.vscode-server/bin/${commit_id} \  
    && cp -r /tmp/vscode-server/*  ~/.vscode-server/bin/${commit_id}/.  

# Install Python requirements
RUN python3 -m pip install debugpy numpy tqdm
COPY ./.docker/requirements_python.txt .
RUN python3 -m pip install -r ./requirements_python.txt

# Cleanup 
RUN apt-get clean autoclean
RUN apt-get autoremove --yes
# Saves ~200MB but DESTROYS APT-GET!
RUN rm -rf /var/lib/{apt,dpkg,cache,log}/

# Creates a non-root user with an explicit UID
ARG USER_NAME="toor"
ARG USER_ID=5678
ARG GROUP_ID=8765
RUN groupadd -g ${GROUP_ID} docker 
RUN useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}
RUN echo "${USER_NAME}:toor" |  chpasswd 
USER $USER_ID:${GROUP_ID}

# Copy VS Code Server to USER
# Note, not necessary, but significant time saver
RUN commit_id=$(cat /tmp/vscode-server/product.json | grep '"commit":' | sed -E 's/.*"([^"]+)".*/\1/') \  
    && mkdir -p ~/.vscode-server/bin/${commit_id} \  
    && cp -r /tmp/vscode-server/*  ~/.vscode-server/bin/${commit_id}/.  

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["python3", "run.py"]
# CMD python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py