#include "mainwindow.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication::setStyle(QStyleFactory::create("fusion"));
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    qRegisterMetaType<const cv::Mat&>("const cv::Mat&");

    return a.exec();
}
