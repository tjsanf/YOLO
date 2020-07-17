sudo docker run \
        -it \
        --rm \
        --gpus all \
        --shm-size 4G \
        --volume $(pwd):$(pwd) \
        --workdir="$(pwd)" \
        tjsanf0606/tensorflow:v0