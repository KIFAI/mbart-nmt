#!/bin/sh
nvidia-docker run -it -d \
	--name translator \
	--gpus all \
        --ipc=host \
        -p 14000:5000 -p 14001:14001 -p 14002:14002 \
	translator:app

