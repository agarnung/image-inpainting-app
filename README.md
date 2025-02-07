# image-inpainting-app
A Qt-based application to restore images with various inpainting methods

<p align="center">
  <img src="./assets/app_main.png" alt="Application main window" title="Application main window" />
</p>

_A work in progress_

The algorithms are implemented in C++ for personal amusement.

# Installation

Make sure you have Qt (https://www.qt.io) installed. Then open the file ```src/ImageInpainting.pro``` with Qt Creator, and build the source code. 

This project uses [OpenCV 4](https://github.com/opencv/opencv/tree/4.10.0) (see additional packages in https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)

The app has been tested on the following platforms:

* Pop!_OS 22.04 LTS with Qt 6.8.1, Qt Creator 15.0.0 and GCC 11.4.0.

# Usage

The app consists of a self-explanatory GUI and it contains a useful Help file. The following table explains the functionality for the GUI icons:

| Icon | Functionality |
|---------- | ---------- |
| ![open](./src/ImageInpainting/icons/open.ico) | Import image |
| ![save](./src/ImageInpainting/icons/save.ico) | Save image |
| ![about](./src/ImageInpainting/icons/about.ico) | Show information about the application |
| ![pencil](./src/ImageInpainting/icons/pencil.ico) | Activate pencil drawing mode (only available in `Noise` image mode) |
| ![pencil_color](./src/ImageInpainting/icons/pencil_color.ico) | Adjust pencil properties |
| `Original image` | Show original image |
| `Noisy image` | Show noisy image |
| `Inpainted image` | Show inpainted image |
| `Inpainting mask` | Show inpainting mask |
| `Clear all` | Clear all images and setup |
| `Reset draw` | Reset pencil drawing |
| `Noise` | Add noise to the image |
| `XYZ Image Inpainting` | Use you favourite image inpainting algorithm |

# TODO
* Mejorar lo de esconder y mostrar las labels. Por ejemplo, unir en un solo el label de mensaje temporal y tipo de imagen.
* Crear GIF de funcionamiento para el README
* Explicaciones de app e iconos en README. Enlazar el README en hipervínculo desde el About de la aplicación
* Exaplicaciones someras con ejemplo visuales de qué algoritmo va mejor para qué caso
* Encapsular toda la app en un Docker
* Release the app in AppImage format
