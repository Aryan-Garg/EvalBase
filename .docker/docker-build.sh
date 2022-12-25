#!/bin/bash 


### CONFIGURE CONTAINER ###

CONTAINER_TAG="eval"
CONTAINER_NAME="aryan_$CONTAINER_TAG" # Suggest "your_name_$CONTAINER_TAG" as a custom container name (can be anything), else it will be random. Cannot have duplicates. 
MOUNT_DIRECTORY_HOST="" # Can be left as "" if not desired. Mounts your project folder.
MOUNT_DIRECTORY_CONTAINER="/workspace" # Where MOUNT_HOST_DIRECTORY will be located in the container
GPU_SELECTION="--gpus all" # Select the GPU hardware to use  --gpus '"device=0,2"' OR --gpus all

# Note! Make sure cuda version matches output from `nvcc --version`
DOCKER_BASE_IMAGE="pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime"
# DOCKER_BASE_IMAGE="nvidia/cuda:11.1.1-base-ubuntu18.04"

# Optional: Mount a project folder
if [[ ! -z $MOUNT_DIRECTORY_HOST && ! -z $MOUNT_DIRECTORY_CONTAINER && -d $MOUNT_DIRECTORY_HOST ]]; then
    MOUNT_DIRECTORY=" -v $MOUNT_DIRECTORY_HOST:$MOUNT_DIRECTORY_CONTAINER"
    echo "Project folder will be mounted -> $MOUNT_DIRECTORY"

    MOUNT_DATASETS=" -v $(pwd)/../datasets/skymangler/unzipped_envmap:/datasets/skymangler/unzipped_envmap"
    echo "Dataset folder will be mounted -> $MOUNT_DATASETS"
    
    MOUNT_DIRECTORY=$MOUNT_DIRECTORY$MOUNT_DATASETS
else 
    echo "Project folder not provided. Skipping."
fi

# Optional: Validate GPU selection
if [[ $(nvidia-smi) ]]; then
    echo "GPU Selection -> $GPU_SELECTION"
else 
    GPU_SELECTION=" "
    echo "Nvidia-smi not available. Skipping GPU selection."
fi

# Optional: Port Forwarding
# For tensorboard use 6006, though note VS Code can forward tensorboard automatically
HOST_PORT="6006"
CONTAINER_PORT="6006"
if [[ $(docker ps) == *":::$HOST_PORT->"* ]]; 
then 
    echo "Host port $HOST_PORT is in use. Skipping." 
else 
    SHARED_PORT="-p $HOST_PORT:$CONTAINER_PORT" # <hostPort:containerPort> (leave as "" for none)
    echo "Port forwaring will be configured -> $SHARED_PORT"
fi

# Optional: SSH Keys
if [ -d "$HOME/.ssh" ]; then
    SSH_CONFIG="-v $HOME/.ssh:/root/.ssh:ro"  # Mound SSH config as a volume (leave as "" for none)
    SSH_CONFIG="$SSH_CONFIG -v $HOME/.ssh:/home/$(whoami)/.ssh:ro"  # Mound SSH config as a volume (leave as "" for none)
    echo "SSH keys will be mounted -> $SSH_CONFIG"
else 
    echo "SSH keys not found. Skipping."
fi

# Optional: Git config
if [ -f "$HOME/.gitconfig" ]; then
    GIT_CONFIG="-v $HOME/.gitconfig:/root/.gitconfig:ro"
    GIT_CONFIG="$GIT_CONFIG -v $HOME/.gitconfig:/home/$(whoami)/.gitconfig:ro"
    echo "Git config will be mounted -> $GIT_CONFIG"
else
    echo "Git config not found. Skipping."
fi

# Optional: XORG
if [ -d "/tmp/.X11-unix" ]; then
    XORG_CONFIG="-v /tmp/.X11-unix:/tmp/.X11-unix:ro"
    echo "XORG config will be mounted -> $XORG_CONFIG"
else
    echo "XORG config not found. Skipping."
fi


### PURGE OLD CONTAINERS ###


# Find container
if [ ! -z $CONTAINER_NAME ]
then
    echo "Checking for an instance of the container $CONTAINER_NAME"
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        # Stop the container
        echo "--> Stopping the container"
        docker stop $CONTAINER_NAME
        if [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
            # cleanup
            echo "--> Removing the container"
            docker rm $CONTAINER_NAME
        fi
    else
        echo "--> No instance of $CONTAINER_NAME found"
    fi
    if [ "$(docker ps -aq -f status=exited)" ]; then
            # cleanup
            echo "--> Prune all stopped containers?"
            docker container prune # Prune all stopped containers
        fi
    echo "Done container purge!"
    CONTAINER_NAME_TO_CREATE="--name $CONTAINER_NAME"
else
    echo "CONTAINER_NAME is blank. Name will be random."
fi

### BUILD CONTAINER ###

# Build the docker image
docker pull $DOCKER_BASE_IMAGE
DOCKER_BUILDKIT=1 docker build --tag  $CONTAINER_TAG --file .docker/Dockerfile --build-arg USER_NAME=$(whoami) --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
if [ $? != 0 ]
then
    echo "<!> Docker build command failed."
    exit
fi 

### START CONTAINER ###

# Start a new container
echo "Starting new container..."
docker run -itd --rm  \
    --ipc=host \
    $GPU_SELECTION \
    $CONTAINER_NAME_TO_CREATE \
    --privileged \
    $MOUNT_DIRECTORY \
    $SHARED_PORT \
    $SSH_CONFIG \
    $GIT_CONFIG \
    --hostname $CONTAINER_NAME \
    $CONTAINER_TAG 

# Note, container is persistent. If not desired, change to:
# ' docker run -it ...'

# Optional: Copy Docker Extensions
if [ -d "$HOME/.vscode-server/extensions" ]; then
    docker cp "$HOME/.vscode-server/extensions" $CONTAINER_NAME:/root/.vscode-server
    docker cp "$HOME/.vscode-server/extensions" "$CONTAINER_NAME:/home/$(whoami)/.vscode-server"
    echo "VS code Extensions copied from -> $HOME/.vscode-server/extensions"
else 
    echo "VS code Extensions not found. Skipping."
fi

echo "Done!"

# Build SlackTools
docker exec -it -w $MOUNT_DIRECTORY_CONTAINER/slackTools $CONTAINER_NAME ./install.sh

# Start tensorboard
docker exec -itd $CONTAINER_NAME tensorboard --logdir logs --bind_all --port 6006 --samples_per_plugin images=999999

# Interact with it
echo "Interact (non-root) with the container using: "
echo "docker exec -it $CONTAINER_NAME <command>"
echo "docker exec -it $CONTAINER_NAME /bin/bash"
echo "docker exec --user $(whoami) -it $CONTAINER_NAME <command>"
echo "docker exec -u francois -it $CONTAINER_NAME <command>"
echo "Or for super user(root): "
echo "docker exec --user root -it $CONTAINER_NAME <command>"
echo "docker exec -u root -it $CONTAINER_NAME /bin/bash"