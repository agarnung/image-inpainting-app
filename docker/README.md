# Introductory Instructions on Docker

> **Note**: Before running the container, on the host machine, the following command should be executed to allow the Docker containers to access the X server, so that graphical applications can be displayed:  
> `$ xhost +local:docker`

# First Steps:
1. Install Docker Compose:  
   `$ sudo apt install docker-compose`
   
2. Verify the installation:  
   `$ docker --version`  
   `Docker version 27.5.1, build 9f9e405`

3. Grant permissions:  
   `$ sudo chmod 777 /var/run/docker.sock`

4. Once the Dockerfile and docker-compose.yml are ready, build the image:  
   `$ docker compose --verbose build`

5. Start the container:  
   `$ docker compose up`

Optionally run `$ sudo ./start.sh`

or...

5. To run the container based on the image you just created, use the following command:  
   `$ docker run -it --name image-inpainting-app-docker-image docker_qt_application`

6. Save the Docker image to a file (Export):  
   `$ docker save -o image-inpainting-app-docker-image.tar docker_qt_application`
   
# Instructions to Run the Container from a `.tar` Image

This project allows running a Qt application inside a Docker container. Follow the steps below to download, load, and run the image.

## **1) Download the Docker Image**
Download the image from the provided link:  
```bash
wget https://your-link.com/my_image.tar
```

## **2) Load the Image into Docker**
Once downloaded, load the image into Docker:
```bash
docker load -i my_image.tar
```
To verify that the image was loaded correctly, run:
```bash
docker images
```
You should see something like this:
```nginx
REPOSITORY                   TAG     IMAGE ID       CREATED        SIZE
qt_image_inpainting_app      latest  abc123def456   2 hours ago    1.2GB
```

## **3) Clone the Repository**
Clone this repository to get the required files (docker-compose.yml and start.sh):
```bash
git clone https://github.com/your-username/image-inpainting-app.git
cd image-inpainting-app
```

## **4) Grant Permissions and Run the Startup Script**
First, ensure that start.sh has execution permissions:
```bash
chmod +x start.sh
```
Then, run the script to launch the container:
```bash
./start.sh
```

## **5) (Optional) View Container Logs**
If you need to check the logs without attaching the terminal, use:
```bash
docker logs -f qt_image_inpainting_app
```
## How to Stop the Container?
To stop and remove the container, run:
```bash
docker-compose down
```

# Troubleshooting:
- Restart the Docker service:  
   `$ sudo systemctl restart docker`

- Check if it is active:  
   `$ sudo systemctl status docker`

- _Error while fetching server API version: Not supported URL scheme http+docker_  
  [Solution](https://stackoverflow.com/questions/64952238/docker-errors-dockerexception-error-while-fetching-server-api-version?page=1&tab=scoredesc#tab-top):  
  `$ pip3 install requests==2.31.0`

- If the docker-compose.yml has syntax errors, the following command will show you:
   `$ docker compose config`

- _Error KeyError: 'ContainerConfig'_ => Remove old (orphnn) containers that may be interfering:
   `$ docker compose down --remove-orphans
      docker system prune -af
   `
# Have you made changes to the source code and want them to reflect in your image?

To ensure that your changes are included in the Docker image, follow these steps:
1. **Compile the application on your host** to make sure everything is up-to-date.
2. Run `docker-compose down` to stop and remove the existing container.
3. Run `docker-compose up -d` to recreate the container. Docker will rebuild the image considering only the changes made to the source code, and then start the container with the updated image.

This process ensures that only the changes you made to the code are included in the new image before running the container again.

# Distribute you image:
Once you have verified that the application works correctly, you can distribute it by uploading the Docker image to Docker Hub or another registry, or distribute it as a file.  

1. Log in to Docker Hub:  
   `$ docker login`

2. Tag the image:  
   `$ docker tag qt_image_inpainting_app <docker_hub_user>/<repository>:<tag>`

3. Upload the image to Docker Hub:  
   `$ docker push <docker_hub_user>/<repository>:<tag>`

4. Now anyone can download and run the application with Docker:  
   `$ docker run -it <docker_hub_user>/<repository>:<tag>`

