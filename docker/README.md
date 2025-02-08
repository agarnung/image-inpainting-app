# Introductory Instructions on Docker

> **Note**: Before running the container, on the host machine, the following  
> command should be executed to allow the Docker containers to access the X server,  
> so that graphical applications can be displayed:  
> `$ xhost +local:docker`

## First Steps:
1. Install Docker Compose:  
   `$ sudo apt install docker-compose`
   
2. Verify the installation:  
   `$ docker --version`  
   `Docker version 27.5.1, build 9f9e405`

3. Grant permissions:  
   `$ sudo chmod 777 /var/run/docker.sock`

4. Once the Dockerfile and docker-compose.yml are ready, build the image:  
   `$ docker-compose build --verbose`

5. Start the container:  
   `$ docker-compose up -d`

or...

5. Para ejecutar el contenedor basado en la imagen que acabas de crear, utiliza el siguiente comando
$ docker run -it --name image-inpainting-app-docker-image docker_qt_application

6. Guardar la imagen de Docker en un archivo (Exportar)
$ docker save -o image-inpainting-app-docker-image.tar docker_qt_application

## Troubleshooting:
- Restart the Docker service:  
   `$ sudo systemctl restart docker`

- Check if it is active:  
   `$ sudo systemctl status docker`

- _Error while fetching server API version: Not supported URL scheme http+docker_  
  [Solution](https://stackoverflow.com/questions/64952238/docker-errors-dockerexception-error-while-fetching-server-api-version?page=1&tab=scoredesc#tab-top):  
  `$ pip3 install requests==2.31.0`

## Distribution:
Once you have verified that the application works correctly, you can distribute it  
by uploading the Docker image to Docker Hub or another registry, or distribute it as a file.  

1. Log in to Docker Hub:  
   `$ docker login`

2. Tag the image:  
   `$ docker tag qt_image_inpainting_app <docker_hub_user>/<repository>:<tag>`

3. Upload the image to Docker Hub:  
   `$ docker push <docker_hub_user>/<repository>:<tag>`

4. Now anyone can download and run the application with Docker:  
   `$ docker run -it <docker_hub_user>/<repository>:<tag>`

