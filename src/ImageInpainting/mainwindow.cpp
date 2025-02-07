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
    mCalculationThread = new CalculationThread(this);
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
    mActionMaxwellHeavisideImageInpainting->setStatusTip("Inpaint image using Maxwell-Heaviside Image Inpainting algorithm");
    QObject::connect(mActionMaxwellHeavisideImageInpainting, &QAction::triggered, this, &MainWindow::showMaxwellHeavisideInpaintingWidget);

    mActionCahnHilliardImageInpainting = new QAction(tr("Cahn-Hilliard Image Inpainting"), this);
    mActionCahnHilliardImageInpainting->setStatusTip("Inpaint image using Cahn-Hilliard Image Inpainting algorithm");
    QObject::connect(mActionCahnHilliardImageInpainting, &QAction::triggered, this, &MainWindow::showCahnHilliardInpaintingWidget);

    mActionBurguersViscousImageInpainting = new QAction(tr("Burgers' Viscous Image Inpainting"), this);
    mActionBurguersViscousImageInpainting->setStatusTip("Inpaint image using Burgers' Viscous Image Inpainting algorithm");
    QObject::connect(mActionBurguersViscousImageInpainting, &QAction::triggered, this, &MainWindow::showBurguersViscousInpaintingWidget);

    mActionTeleaImageInpainting = new QAction(tr("Telea Image Inpainting"), this);
    mActionTeleaImageInpainting->setStatusTip("Inpaint image using Telea Image Inpainting algorithm");
    QObject::connect(mActionTeleaImageInpainting, &QAction::triggered, this, &MainWindow::showTeleaInpaintingWidget);

    mActionNavierStokesImageInpainting = new QAction(tr("Navier-Stokes Image Inpainting"), this);
    mActionNavierStokesImageInpainting->setStatusTip("Inpaint image using Navier-Stokes Image Inpainting algorithm");
    QObject::connect(mActionNavierStokesImageInpainting, &QAction::triggered, this, &MainWindow::showNavierStokesInpaintingWidget);

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
    mActionClearImage->setText("Clear all");
    mActionClearImage->setStatusTip("Clear all");
    QObject::connect(mActionClearImage, &QAction::triggered, this, &MainWindow::clearImage);

    mActionResetDraw = new QAction(this);
    mActionResetDraw->setText("Reset draw");
    mActionResetDraw->setStatusTip("Reset draw");
    QObject::connect(mActionResetDraw, &QAction::triggered, this, &MainWindow::resetDraw);
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
    mMenuAlgorithms->addAction(mActionCahnHilliardImageInpainting);
    mMenuAlgorithms->addAction(mActionBurguersViscousImageInpainting);
    mMenuAlgorithms->addAction(mActionTeleaImageInpainting);
    mMenuAlgorithms->addAction(mActionNavierStokesImageInpainting);
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
    mToolbarImageStatus->addAction(mActionResetDraw);
    mToolbarImageStatus->addAction(mActionClearImage);
    mToolbarImageStatus->setEnabled(false);
}

void MainWindow::createStatusBar()
{
    mLabelOperationInfo = new QLabel(this);
    mLabelOperationInfo->setAlignment(Qt::AlignCenter);
    mLabelOperationInfo->setMinimumSize(mLabelOperationInfo->sizeHint());

    mLabelOtherInfo = new QLabel(this);
    mLabelOtherInfo->setAlignment(Qt::AlignLeft);
    mLabelOtherInfo->setMinimumSize(mLabelOtherInfo->sizeHint());

    QWidget* statusWidget = new QWidget(this);

    QVBoxLayout* layout = new QVBoxLayout(statusWidget);

    layout->addWidget(mLabelOperationInfo);
    layout->addWidget(mLabelOtherInfo);

    statusWidget->setLayout(layout);
    statusWidget->setMinimumSize(statusWidget->sizeHint());
    statusBar()->addWidget(statusWidget);

    QObject::connect(mIOThread, &IOThread::statusShowMessage, mLabelOperationInfo, &QLabel::setText);
    QObject::connect(mCalculationThread, &CalculationThread::statusShowMessage, mLabelOperationInfo, &QLabel::setText);
    QObject::connect(mDataManager, &DataManager::statusShowMessage, mLabelOtherInfo, &QLabel::setText);
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

void MainWindow::receiveProcessImage(const cv::Mat& img)
{
    if (!mDataManager || !mImageViewer)
        return;

    if (img.depth() != CV_8U)
    {
        cv::Mat img_8U;
        img.convertTo(img_8U, CV_8U, 255.0);
        mDataManager->setInpaintedImage(img_8U);
    }
    else
        mDataManager->setInpaintedImage(img);

    transToInpaintedImage();
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

    if (!filename.contains('.'))
        filename.append(".png");

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

void MainWindow::resetDraw()
{
    mImageViewer->clearDrawing();
    mDataManager->setMask(cv::Mat(mDataManager->getMask().size(), CV_8UC1, cv::Scalar(255)));
}

void MainWindow::applyAlgorithm(QString algorithmName)
{
    mActionToInpaintedImage->setChecked(true);
    transToInpaintedImage();
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

void MainWindow::showCahnHilliardInpaintingWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kCahnHilliardImageInpainting;
    closeWidget();
    showWidget();
}

void MainWindow::showBurguersViscousInpaintingWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kBurgersViscousImageInpainting;
    closeWidget();
    showWidget();
}

void MainWindow::showTeleaInpaintingWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kTeleaImageInpainting;
    closeWidget();
    showWidget();
}

void MainWindow::showNavierStokesInpaintingWidget()
{
    mCalculationThread->mAlgorithmType = CalculationThread::kNavierStokesImageInpainting;
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

