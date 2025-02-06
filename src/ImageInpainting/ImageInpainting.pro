QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageInpainting
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
        Algorithms/Noise.cpp \
        Algorithms/ImageInpaintingBase.cpp \
        Algorithms/MaxwellHeavisideImageInpainting.cpp \
        Algorithms/TeleaImageInpainting.cpp \

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
        Algorithms/TeleaImageInpainting.h \

RESOURCES += \
    icons.qrc

CONFIG += link_pkgconfig
PKGCONFIG += opencv4
