#!/bin/bash

xhost +local:docker

mkdir -p data

sudo chown $USER:$USER data
chmod 777 data 

docker-compose down --remove-orphans

docker-compose up -d 

docker attach qt_image_inpainting_app

