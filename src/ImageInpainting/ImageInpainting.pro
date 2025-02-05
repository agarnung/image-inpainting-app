QT += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageInpainting
TEMPLATE = app

CONFIG += c++20

SOURCES += \
        main.cpp \
        mainwindow.cpp

FORMS += \
        mainwindow.ui

HEADERS += \
        mainwindow.h \
        utils.h

RESOURCES += \
    icons.qrc
