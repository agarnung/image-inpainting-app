# Usamos Ubuntu 24.04 como base
FROM ubuntu:24.04

# Configurar zona horaria 
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Actualizar repositorios
RUN apt-get update && apt-get install -y \
    build-essential \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-qmake \
    qt6-doc \
    qt6-dev-tools \
    qt6-quickcontrols2 \
    qt6-serialport \
    libxcb1 libx11-6 libxext6 libxi6 libxrender1 libfontconfig1 \
    libfreetype6 libglu1-mesa \
    libx11-xcb1 \
    libgl1-mesa-glx \
    x11-utils \
    xorg-dev \
    wget \
    git \
    cmake \
    libopencv-dev \
    libopencv-gapi-dev \
    libopencv-stitching-dev \
    libopencv-aruco-dev \
    libopencv-bgsegm-dev \
    libopencv-bioinspired-dev \
    libopencv-ccalib-dev \
    libopencv-dnn-dev \
    libopencv-dnn-superres-dev \
    libopencv-face-dev \
    libopencv-ffmpeg-dev \
    libopencv-gpu-dev \
    libopencv-imgproc-dev \
    libopencv-core-dev \
    libopencv-imgcodecs-dev \
    libopencv-photo-dev \
    libopencv-videoio-dev \
    libopencv-video-dev \
    libopencv-objdetect-dev \
    libopencv-features2d-dev \
    libopencv-calib3d-dev \
    libopencv-highgui-dev \
    libopencv-flann-dev \
    libopencv-xfeatures2d-dev \
    libopencv-ximgproc-dev \
    libopencv-datasets-dev \
    libopencv-text-dev \
    && apt-get clean

# Crear el directorio de trabajo
WORKDIR /app

# Crear un usuario para evitar usar root
RUN useradd -ms /bin/bash appuser
USER appuser

# Copiar en el directorio de trabajo el código fuente del proyecto
COPY --chown=appuser:appuser ./src /app

# Establecer la ruta a Qt en el contenedor (especificar una instalación estándar de Qt dentro del contenedor)
ENV QTDIR=/usr/lib/qt6
ENV PATH=$QTDIR/bin:$PATH
ENV LD_LIBRARY_PATH=$QTDIR/lib:$LD_LIBRARY_PATH

# Ejecutar el comando qmake para generar los Makefiles
RUN qmake /app/ImageInpainting.pro

# Compilar la aplicación usando make
RUN make -C /app

# Exponer un puerto por si es útil a futuro (e.g. para servidor web)
EXPOSE 8080

# Configurar el comando para ejecutar la aplicación Qt
CMD ["./ImageInpainting"]

