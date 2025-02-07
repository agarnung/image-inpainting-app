# image-inpainting-app
A Qt-based application to restore images with various inpainting methods

<p align="center">
  <img src="./assets/app_main.png" alt="Application main window" title="Application main window" />
</p>

The algorithms are implemented in C++ for personal amusement.

# Installation

Make sure you have Qt (https://www.qt.io) installed. Then open the file ```src/MeshDenoising.pro``` with Qt Creator, and build the source code. 

This project uses [OpenCV 4](https://github.com/opencv/opencv/tree/4.10.0) (see additional packages in https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)

The app has been tested on the following platforms:

* Pop!_OS 22.04 LTS with Qt 6.8.1, Qt Creator 15.0.0 and GCC 11.4.0.

# Usage

The app consists of a self-explanatory GUI and it contains a useful Help file. The following table explains the functionality for the GUI icons:

|Icon       | Functionality |
|---------- | ----------    |
|![name](./path/to/qrc/icon.ico) | Explanation |
|![name](./path/to/qrc/icon.ico) | Explanation |
|`name` | Explanation |

# TODO
* Poner en otro label "procesando canal R, G, B o gris durante las iteraciones..."
* Arreglar el por qué no se muestra la imagen resultado al terminal el algoritmo, por qué se necesita pulsar el botón InpaintedImage para que se recargue
* Ver cómo lanzar mensajes temporales a los labels del statusBar con un QTimer
* Arreglar el hecho de que la originalImage se sobreescriba con la noisyImage en la segunda ejecución
* Crear GIF de funcionamiento para el README
* Explicaciones de app e iconos en README. Enlazar el README en hipervínculo desde el About de la aplicación
* Encapsular toda la app en un Docker
* Release the app in AppImage format
