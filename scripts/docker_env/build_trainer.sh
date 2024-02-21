#!/bin/bash

# CUDA version
cuda_version="11.4.3"
# Ubuntu version
ubuntu_version="20.04"
# Container name
container_name="trainer"

# Check if docker image exists
image_name="nvidia/cuda:$cuda_version-devel-ubuntu$ubuntu_version"
if docker image inspect "$image_name" &> /dev/null; then
    echo "Docker image $image_name already exists."
else
    echo "Docker image $image_name does not exist. Pulling..."
    docker pull "$image_name"
fi

docker run -it -d --name "$container_name" --network host --gpus all --ipc=host "$image_name"

sleep 5

docker cp set_py38_env.sh "$container_name":/opt

# Run set_py38_env.sh inside the container
docker exec -it trainer /bin/bash -c '/opt/set_py38_env.sh'