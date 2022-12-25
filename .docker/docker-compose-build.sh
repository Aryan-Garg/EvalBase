#!/bin/bash

# Optional: Validate GPU selection
COMPOSE_SELECTION="./.docker/docker-compose.yml"
if [[ $(nvidia-smi) ]]; then
    echo "Nvidia GPUs available! "
    COMPOSE_SELECTION="./.docker/docker-compose.gpu.yml"
else 
    echo "Nvidia-smi not available. Skipping GPU selection."
fi

# Build
docker-compose -f $COMPOSE_SELECTION build --build-arg USER_NAME=$(whoami) --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) skynet
    
# # Run
# docker-compose -f $COMPOSE_SELECTION up -d --no-recreate

echo done!

# Interact with it
echo "Interact (non-root) with the container using: "
echo "docker exec -it $CONTAINER_NAME <command>"
echo "docker exec -it $CONTAINER_NAME /bin/bash"
echo "docker exec --user $(whoami) -it $CONTAINER_NAME <command>"
echo "docker exec --u francois -it $CONTAINER_NAME <command>"
echo "Or for super user(root): "
echo "docker exec --user root -it $CONTAINER_NAME <command>"
echo "docker exec --u root -it $CONTAINER_NAME /bin/bash"