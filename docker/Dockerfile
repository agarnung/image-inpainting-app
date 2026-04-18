# syntax=docker/dockerfile:1
# BuildKit required: DOCKER_BUILDKIT=1 docker compose build
#
# Multi-stage Qt6 + OpenCV build. Uses Ubuntu's libopencv-dev (OpenCV 4.6),
# which covers every module the app imports (core, imgproc, imgcodecs, photo,
# highgui). Building OpenCV from source took ~30 min for no benefit.

FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/lib/ccache:${PATH}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ccache \
        pkg-config \
        qt6-base-dev \
        qt6-base-dev-tools \
        qmake6 \
        libgl1-mesa-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY src/ImageInpainting/ ./

RUN --mount=type=cache,target=/root/.ccache \
    mkdir -p build \
    && cd build \
    && qmake6 CONFIG+=release ../ImageInpainting.pro \
    && make -j"$(nproc)" \
    && install -m755 ImageInpaintingApp /usr/local/bin/ImageInpaintingApp

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libqt6widgets6t64 \
        libqt6gui6t64 \
        libqt6core6t64 \
        libqt6svg6 \
        libqt6dbus6t64 \
        qt6-qpa-plugins \
        libglib2.0-0 \
        libfontconfig1 \
        libfreetype6 \
        libxcb-xinerama0 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libxcb1 \
        libx11-6 \
        libx11-xcb1 \
        libsm6 \
        libice6 \
        libgl1 \
        libopencv-core406t64 \
        libopencv-imgproc406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-photo406t64 \
        libopencv-highgui406t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/ImageInpaintingApp /usr/local/bin/ImageInpaintingApp
RUN ! ldd /usr/local/bin/ImageInpaintingApp | grep -q 'not found'

WORKDIR /data
ENV QT_QPA_PLATFORM=xcb \
    APP_IMAGES_DIR=/data
ENTRYPOINT ["/usr/local/bin/ImageInpaintingApp"]
