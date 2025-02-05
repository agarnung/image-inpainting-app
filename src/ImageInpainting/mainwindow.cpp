#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "calculationthread.h"
#include "datamanager.h"
#include "imageviewer.h"
#include "iothread.h"
#include "parameterset.h"
#include "parametersetwidget.h"
#include "utils.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , mUi(new Ui::MainWindow)
{
    mUi->setupUi(this);

    setWindowTitle("ImageInpainting App");
    setWindowIcon(QIcon(":/icons/pencil.ico"));

    setGeometry(250, 50, 880, 660);
}

MainWindow::~MainWindow()
{
    DELETE(mUi)
}

void MainWindow::init()
{
    mImageViewer = new ImageViewer();
    mDataManager = new DataManager();
    mParameterSet = new ParameterSet();
    mParameterSetWidget = new ParameterSetWidget();
    mCalculationThread = new CalculationThread();
    mCalculationThread->mAlgorithmType = CalculationThread::kNone;
    connect(mCalculationThread, &CalculationThread::needToUpdate, this, &MainWindow::needToUpdate);
    QObject::connect(mCalculationThread, &CalculationThread::setActionAndWidget, this, &MainWindow::setActionAndWidget);
    mIOThread = new IOThread(mDataManager);
    mIOThread->mIOType = IOThread::kNone;
    QObject::connect(mIOThread, &IOThread::needToUpdate, this, &MainWindow::needToUpdate);
    QObject::connect(mIOThread, &IOThread::setActionAndWidget, this, &MainWindow::setActionAndWidget);
}

void MainWindow::setActionAndWidget(bool value1, bool value2)
{
    ;
}

void MainWindow::needToUpdate(bool value)
{
    ;
}

void MainWindow::About()
{
    QMessageBox::about(this, tr("About ImageInpainting App"),
                       tr("Here goes the info. "
                          "May add HTML and CSS in a Scrollable pop-up with instruction."));
}

