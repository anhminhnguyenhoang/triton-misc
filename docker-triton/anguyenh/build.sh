cd ~/docker-triton/anguyenh/
 
# generate all necessary environment variables
# the variables are going to be stored
# in the same directory in `.env` file
./gen-env-file.sh
 
# build the image and spin a container
docker compose build --no-cache
 
# Note, starting from docker v26, `compose` became an internal docker command.
# Used `docker-compose` if you have an older version of docker.
 
# if you want to spin more than one container
# - e.g., for debugging purposes - you can do it as follows
# docker compose up -d --scale triton_dev=2
# where `triton_dev` is the name of the service described
# in the docker-compose.yaml file (see, below)
 
# inspect running containers
docker ps
 
# jump into a running container - e.g.,
# docker exec -it jukorhon-triton-1 /bin/bash