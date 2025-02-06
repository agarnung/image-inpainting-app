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

    setStyle();

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
    mDataManager = new DataManager();
    mImageViewer = new ImageViewer(nullptr, mDataManager);
    mParameterSet = new ParameterSet();
    mParameterSetWidget = new ParameterSetWidget();
    mCalculationThread = new CalculationThread();
    mCalculationThread->mAlgorithmType = CalculationThread::kNone;
    QObject::connect(mCalculationThread, &CalculationThread::needToResetImage, this, &MainWindow::needToResetImage);
    QObject::connect(mCalculationThread, &CalculationThread::setActionAndWidget, this, &MainWindow::setActionAndWidget);
    mIOThread = new IOThread(mDataManager);
    mIOThread->mIOType = IOThread::kNone;
    QObject::connect(mIOThread, &IOThread::needToResetImage, this, &MainWindow::needToResetImage);
    QObject::connect(mIOThread, &IOThread::setActionAndWidget, this, &MainWindow::setActionAndWidget);
}

void MainWindow::createActions()
{
    mActionImportImage = new QAction(QIcon(":/icons/open.ico"), tr("Import image"), this);
    mActionImportImage->setStatusTip("Import image");
    QObject::connect(mActionImportImage, &QAction::triggered, this, &MainWindow::importImage);

    mActionExportImage = new QAction(QIcon(":/icons/save.ico"), tr("Save image"), this);
    mActionExportImage->setStatusTip("Save image");
    QObject::connect(mActionExportImage, &QAction::triggered, this, &MainWindow::exportImage);

    mActionExit = new QAction(tr("Exit"), this);
    mActionExit->setStatusTip("Exit");
    QObject::connect(mActionExit, &QAction::triggered, this, &MainWindow::close);

    mActionNoise = new QAction(tr("Noise"), this);
    mActionNoise->setStatusTip("Add noise to image using 'algorithm' -- Noise");
    QObject::connect(mActionNoise, &QAction::triggered, this, &MainWindow::showNoiseWidget);

    mActionMaxwellHeavisideImageInpainting = new QAction(tr("Maxwell-Heaviside Image Inpainting"), this);
    mActionMaxwellHeavisideImageInpainting->setStatusTip("Inpaint image using algorithm -- Maxwell-Heaviside Image Inpainting");
    QObject::connect(mActionMaxwellHeavisideImageInpainting, &QAction::triggered, this, &MainWindow::showMaxwellHeavisideInpaintingWidget);

    mActionTeleaImageInpainting = new QAction(tr("Telea Image Inpainting"), this);
    mActionTeleaImageInpainting->setStatusTip("Inpaint image using algorithm -- Telea Image Inpainting");
    QObject::connect(mActionTeleaImageInpainting, &QAction::triggered, this, &MainWindow::showTeleaInpaintingWidget);

    mActionAbout = new QAction(QIcon(":/icons/about.ico"), tr("About"), this);
    mActionAbout->setStatusTip("Information about this application");
    QObject::connect(mActionAbout, &QAction::triggered, this, &MainWindow::about);

    mActionRenderPencil = new QAction(tr("Draw mode"), this);
    mActionRenderPencil->setStatusTip("Enable or disable the pencil");
    mActionRenderPencil->setCheckable(true);
    QObject::connect(mActionRenderPencil, &QAction::triggered, mImageViewer, &ImageViewer::togglePencilDrawing);

    mActionPencilProperties = new QAction(QIcon(":/icons/pencil_color.ico"), tr("Pencil properties"), this);
    mActionPencilProperties->setStatusTip("Specify the pencil properties");
    QObject::connect(mActionPencilProperties, &QAction::triggered, mImageViewer, &ImageViewer::showPencilSettingsDialog);

    QActionGroup* actionGroup = new QActionGroup(this);
    actionGroup->setExclusive(true);

    mActionToOriginalImage = new QAction(this);
    mActionToOriginalImage->setText("Original image");
    mActionToOriginalImage->setStatusTip("Show original image");
    mActionToOriginalImage->setCheckable(true);
    QObject::connect(mActionToOriginalImage, &QAction::triggered, this, &MainWindow::transToOriginalImage);
    actionGroup->addAction(mActionToOriginalImage);

    mActionToNoisyImage = new QAction(this);
    mActionToNoisyImage->setText("Noisy image");
    mActionToNoisyImage->setStatusTip("Show noisy image");
    mActionToNoisyImage->setCheckable(true);
    QObject::connect(mActionToNoisyImage, &QAction::triggered, this, &MainWindow::transToNoisyImage);
    actionGroup->addAction(mActionToNoisyImage);

    mActionToInpaintedImage = new QAction(this);
    mActionToInpaintedImage->setText("Inpainted image");
    mActionToInpaintedImage->setStatusTip("Show inpainted image");
    mActionToInpaintedImage->setCheckable(true);
    QObject::connect(mActionToInpaintedImage, &QAction::triggered, this, &MainWindow::transToInpaintedImage);
    actionGroup->addAction(mActionToInpaintedImage);

    mActionToMask = new QAction(this);
    mActionToMask->setText("Inpainting mask");
    mActionToMask->setStatusTip("Show inpainting mask");
    mActionToMask->setCheckable(true);
    QObject::connect(mActionToMask, &QAction::triggered, this, &MainWindow::transToMask);
    actionGroup->addAction(mActionToMask);

    mActionClearImage = new QAction(this);
    mActionClearImage->setText("Clear");
    mActionClearImage->setStatusTip("Clear image");
    QObject::connect(mActionClearImage, &QAction::triggered, this, &MainWindow::clearImage);
}

void MainWindow::createMenus()
{
    mMenuFile = menuBar()->addMenu(tr("File"));
    mMenuFile->addAction(mActionImportImage);
    mMenuFile->addAction(mActionExportImage);
    mMenuFile->addSeparator();
    mMenuFile->addAction(mActionExit);

    mMenuAlgorithms = menuBar()->addMenu(tr("Algorithms"));
    mMenuAlgorithms->addAction(mActionNoise);
    mMenuAlgorithms->addSeparator();
    mMenuAlgorithms->addAction(mActionMaxwellHeavisideImageInpainting);
    mMenuAlgorithms->addAction(mActionTeleaImageInpainting);
    mMenuAlgorithms->setEnabled(false);

    mMenuHelp = menuBar()->addMenu(tr("Help"));
    mMenuHelp->addAction(mActionAbout);
}

void MainWindow::createToolBars()
{
    mToolbarFile = addToolBar(tr("File"));
    mToolbarFile->addAction(mActionImportImage);
    mToolbarFile->addAction(mActionExportImage);

    mToolbarDrawInfo = addToolBar(tr("Draw"));
    mToolbarDrawInfo->addAction(mActionRenderPencil);
    mToolbarDrawInfo->addSeparator();
    mToolbarDrawInfo->addAction(mActionPencilProperties);

    mToolbarImageStatus = addToolBar(tr("Status"));
    mToolbarImageStatus->addAction(mActionToOriginalImage);
    mToolbarImageStatus->addAction(mActionToNoisyImage);
    mToolbarImageStatus->addAction(mActionToInpaintedImage);
    mToolbarImageStatus->addAction(mActionToMask);
    mToolbarImageStatus->addSeparator();
    mToolbarImageStatus->addAction(mActionClearImage);
    mToolbarImageStatus->setEnabled(false);
}

void MainWindow::createStatusBar()
{
    mLabelOperationInfo = new QLabel();
    mLabelOperationInfo->setAlignment(Qt::AlignCenter);
    mLabelOperationInfo->setMinimumSize(mLabelOperationInfo->sizeHint());

    statusBar()->addWidget(mLabelOperationInfo);
    connect(mIOThread, &IOThread::statusShowMessage, mLabelOperationInfo, &QLabel::setText);
    connect(mCalculationThread, &CalculationThread::statusShowMessage, mLabelOperationInfo, &QLabel::setText);
}

void MainWindow::setActionStatus(bool value)
{
    mActionImportImage->setEnabled(value);
    mActionExportImage->setEnabled(value);

    mMenuAlgorithms->setEnabled(value);

    mToolbarImageStatus->setEnabled(value);
}

void MainWindow::closeWidget()
{
    DELETE(mParameterSetWidget)
}

void MainWindow::showWidget()
{
    mCalculationThread->initAlgorithm(mDataManager, mParameterSet);
    mParameterSetWidget = new ParameterSetWidget(this, mParameterSet);

    QObject::connect(mParameterSetWidget, &ParameterSetWidget::readyToApply, this, &MainWindow::applyAlgorithm);
    addDockWidget(Qt::RightDockWidgetArea, mParameterSetWidget);
    mParameterSetWidget->showWidget();
}

void MainWindow::setStyle()
{
    QFile styleFile(":/style.qss");
    if (styleFile.open(QFile::ReadOnly | QFile::Text))
    {
        QTextStream stream(&styleFile);
        QString styleSheet = stream.readAll();
        this->setStyleSheet(styleSheet);

        styleFile.close();
    }
    else
        qCritical() << QObject::tr("%1 - %2 - Could not open style sheet").arg(this->metaObject()->className()).arg(__func__);
}

void MainWindow::importImage()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Import image"), ".", tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"));

    if(filename.isNull()) return;

    mIOThread->setFileName(filename);
    mIOThread->mIOType = IOThread::kImport;
    mIOThread->start();
}

void MainWindow::exportImage()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Export image"), ".", tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"));

    if(filename.isNull()) return;

    mIOThread->setFileName(filename);
    mIOThread->mIOType = IOThread::kExport;
    mIOThread->start();
}

void MainWindow::transToNoisyImage()
{
    mDataManager->imageToNoisyImage();
    mImageViewer->resetImage(mDataManager->getImagePixmap());
    mImageViewer->update();
}

void MainWindow::transToOriginalImage()
{
    mDataManager->imageToOriginalImage();
    mImageViewer->resetImage(mDataManager->getImagePixmap());
    mImageViewer->update();
}

void MainWindow::transToInpaintedImage()
{
    mDataManager->imageToInpaintedImage();
    mImageViewer->resetImage(mDataManager->getImagePixmap());
    mImageViewer->update();
}

void MainWindow::transToMask()
{
    mDataManager->imageToMask();
    mImageViewer->resetImage(mDataManager->getMaskPixmap());
    mImageViewer->update();
}

void MainWindow::clearImage()
{
    closeWidget();
    mDataManager->clearImage();
    mImageViewer->resetImage(mDataManager->getImagePixmap());
    mImageViewer->clearDrawing();
    mImageViewer->update();
    setActionStatus(false);
    mActionImportImage->setEnabled(true);
}

void MainWindow::applyAlgorithm(QString algorithmName)
{
    mCalculationThread->setAlgorithmName(algorithmName);
    mCalculationThread->start();
}

void MainWindow::showNoiseWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kNoise;
    closeWidget();
    showWidget();
}

void MainWindow::showMaxwellHeavisideInpaintingWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kMaxwellHeavisideImageInpainting;
    closeWidget();
    showWidget();
}

void MainWindow::showTeleaInpaintingWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kTeleaImageInpainting;
    closeWidget();
    showWidget();
}

void MainWindow::setActionAndWidget(bool value1, bool value2)
{
    setActionStatus(value1);

    if (mParameterSetWidget && value2)
    {
        closeWidget();
        showWidget();
    }
}

void MainWindow::needToResetImage()
{
    mImageViewer->resetImage(mDataManager->getImagePixmap());
    mImageViewer->update();
}

void MainWindow::about()
{
    QMessageBox::about(this, tr("About ImageInpainting App"),
                       tr("Here goes the info. <br>"
                          "May add HTML and CSS in a Scrollable pop-up with instruction."));
}

