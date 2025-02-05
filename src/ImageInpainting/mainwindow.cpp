#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "utils.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    setWindowTitle("ImageInpainting UI");
    setWindowIcon(QIcon(":/icons/pencil.ico"));

    setGeometry(250, 50, 880, 660);
}

MainWindow::~MainWindow()
{
    DELETE(ui)
}
