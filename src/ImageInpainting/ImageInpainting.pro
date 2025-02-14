QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageInpaintingApp
TEMPLATE = app

CONFIG += c++20

SOURCES += \
        calculationthread.cpp \
        datamanager.cpp \
        imageviewer.cpp \
        iothread.cpp \
        main.cpp \
        mainwindow.cpp \
        parameterset.cpp \
        parametersetwidget.cpp \
        utils.cpp \
        Algorithms/Noise.cpp \
        Algorithms/ImageInpaintingBase.cpp \
        Algorithms/MaxwellHeavisideImageInpainting.cpp \
        Algorithms/BurgersViscousImageInpainting.cpp \
        Algorithms/CahnHilliardImageInpainting.cpp \
        Algorithms/NavierStokesImageInpainting.cpp \
        Algorithms/TeleaImageInpainting.cpp \
        Algorithms/CriminisiImageInpainting.cpp \
        Algorithms/FastDigitalImageInpainting.cpp \

FORMS += \
        mainwindow.ui

HEADERS += \
        calculationthread.h \
        datamanager.h \
        imageviewer.h \
        iothread.h \
        mainwindow.h \
        parameterset.h \
        parametersetwidget.h \
        utils.h \
        Algorithms/Noise.h \
        Algorithms/ImageInpaintingBase.h \
        Algorithms/MaxwellHeavisideImageInpainting.h \
        Algorithms/BurgersViscousImageInpainting.h \
        Algorithms/CahnHilliardImageInpainting.h \
        Algorithms/NavierStokesImageInpainting.h \
        Algorithms/TeleaImageInpainting.h \
        Algorithms/CriminisiImageInpainting.h \
        Algorithms/FastDigitalImageInpainting.h \

RESOURCES += \
    icons.qrc

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

QMAKE_CXXFLAGS += -O3 -march=native
