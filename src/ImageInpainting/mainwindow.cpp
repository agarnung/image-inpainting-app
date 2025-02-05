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

    init();
    createActions();
    createMenus();
    createToolBars();
    createStatusBar();

    QVBoxLayout* layoutLeft = new QVBoxLayout;
    layoutLeft->addStretch(2);

    QHBoxLayout* layoutMain = new QHBoxLayout;
    layoutMain->addLayout(layoutLeft);
    layoutMain->addWidget(mImageViewer);
    layoutMain->setStretch(1, 1);
    this->centralWidget()->setLayout(layoutMain);
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

void MainWindow::createActions()
{
    mActionImportImage = new QAction(QIcon(":/icons/open.ico"),tr("Import image"),this);
    mActionImportImage->setStatusTip("Import image");
    QObject::connect(mActionImportImage, &QAction::triggered, this, &MainWindow::ImportImage);

    //...
}

void MainWindow::createMenus()
{
    ;
}

void MainWindow::createToolBars()
{
    mToolbarFile = addToolBar(tr("File"));
    mToolbarFile->addAction(mActionImportImage);

    //...
}

void MainWindow::createStatusBar()
{
    ;
}

void MainWindow::ImportImage()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Import image"), ".", tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"));

    if(filename.isNull()) return;

    mIOThread->setFileName(filename);
    mIOThread->mIOType = IOThread::kImport;
    mIOThread->start();
}

void MainWindow::ExportImage()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Export image"), ".", tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"));

    if(filename.isNull()) return;

    mIOThread->setFileName(filename);
    mIOThread->mIOType = IOThread::kExport;
    mIOThread->start();
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

