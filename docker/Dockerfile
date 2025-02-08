###########################
# Stage 1: Compile OpenCV #
###########################
# Build OpenCV (this stage is cached, so it won't be rebuilt unless changed)
FROM ubuntu:24.04 AS opencv-build

# Update repositories
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-declarative-dev \
    qt6-multimedia-dev \
    qt6-positioning-dev \
    qt6-webengine-dev \
    qt6-svg-dev \
    libxcb1 libx11-6 libxext6 libxi6 libxrender1 libfontconfig1 \
    libfreetype6 libglu1-mesa \
    x11-utils \
    libopenexr-dev \
    openexr \
    xorg-dev \
    wget \
    git \
    cmake \
    pkg-config \
    libxcb-xinerama0 \
    libx11-xcb1 \
    xvfb \
    libglu1-mesa \
    libxcb-xinerama0 \
    libjpeg-dev \
    libopencv-dev \
    libpng-dev \
    libtiff-dev \
    libdcmtk-dev \
    libeigen3-dev \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create the working directory
WORKDIR /app

# Create a user to avoid running as root
RUN useradd -ms /bin/bash appuser

# Configure time zone
USER root
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# Clone OpenCV and opencv_contrib repositories
WORKDIR /opt
RUN git config --global http.postBuffer 524288000 # Increases the buffer size to 512 MB to prevent disconnection errors.
RUN git config --global core.compression 0
RUN git clone --depth=1 --branch 4.11.0 https://github.com/opencv/opencv.git -c http.extraHeader="X-Forwarded-For: 192.168.1.1"
RUN git clone --depth=1 --branch 4.11.0 https://github.com/opencv/opencv_contrib.git -c http.extraHeader="X-Forwarded-For: 192.168.1.1"

# Switch to the OpenCV directory and checkout the desired version
WORKDIR /opt/opencv
RUN git checkout 4.11.0 # Choose version 4.x of OpenCV

WORKDIR /opt/opencv_contrib
RUN git checkout 4.11.0 # Choose version 4.x of opencv_contrib

# Create a directory for the compilation
WORKDIR /opt/opencv
RUN mkdir build
WORKDIR build

# Configure the OpenCV build
# https://github.com/openframeworks/openFrameworks/issues/4576
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local/ \
    -D CMAKE_INSTALL_LIBDIR=lib \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_LIST=core,imgproc,imgcodecs,highgui,photo \
    -D WITH_OPENEXR=OFF \ 
#    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules/ \
#    -D WITH_QT=ON \
#    -D WITH_OPENGL=ON \
    ..

# Compile OpenCV
RUN make -j$(nproc) # Use multiple cores to speed up the compilation process
RUN make install

# Clean up the build directory to free up space
WORKDIR /opt
RUN rm -rf opencv opencv_contrib

# If the pkg-config tool isn't able to find it, it was not properly installed
RUN ls /usr/local/lib/pkgconfig | grep opencv4

###########################
# Stage 2: Qt application #
###########################
# Application build (OpenCV is already installed, so no need to recompile it)
FROM ubuntu:24.04

# Install necessary dependencies for the application
RUN apt-get update && apt-get install -y \
    g++ \
    qt6-base-dev \
    qt6-tools-dev \
    qt6-declarative-dev \
    qt6-multimedia-dev \
    qt6-positioning-dev \
    qt6-webengine-dev \
    qt6-svg-dev \
    libxcb1 libx11-6 libxext6 libxi6 libxrender1 libfontconfig1 \
    libfreetype6 libglu1-mesa \
    libx11-xcb1 \
    x11-utils \
    xorg-dev \
    wget \
    git \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdcmtk-dev \
    libeigen3-dev \
    libatlas-base-dev \
    && apt-get clean

# Copy the already compiled OpenCV artifacts from the previous stage
COPY --from=opencv-build /usr/local /usr/local

# Update the linker cache so OpenCV shared libraries are found
RUN ldconfig

# Set Qt path within the container (specifying a standard Qt installation inside the container)
ENV QTDIR=/usr/lib/qt6
ENV PATH=$QTDIR/bin:$PATH
ENV LD_LIBRARY_PATH=$QTDIR/lib:/usr/local/lib:$LD_LIBRARY_PATH

# Set environment variables for OpenCV
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/usr/lib/qt6/lib:/usr/local/lib:$LD_LIBRARY_PATH

# Create the working directory and copy the project source code into the working directory
WORKDIR /app
COPY --chown=appuser:appuser ../src /app

# Create a user to avoid running as root
RUN useradd -ms /bin/bash appuser

# Fix permissions to ensure appuser has access to necessary files
RUN chown -R appuser:appuser /opt
RUN chmod -R 755 /opt
RUN chmod -R 777 /app

USER appuser

# Run qmake to generate the Makefiles
RUN qmake /app/ImageInpainting/ImageInpainting.pro

# Compile the application using make
RUN make -C /app

# Expose a port in case it's useful in the future (e.g., for a web server)
EXPOSE 8080

# Configure the command to run the Qt application
CMD ["./ImageInpaintingApp"]

