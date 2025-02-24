#include "calculationthread.h"
#include "utils.h"
#include "Algorithms/ImageInpaintingBase.h"
#include "Algorithms/TeleaImageInpainting.h"
#include "Algorithms/NavierStokesImageInpainting.h"
#include "Algorithms/MaxwellHeavisideImageInpainting.h"
#include "Algorithms/CahnHilliardImageInpainting.h"
#include "Algorithms/BurgersViscousImageInpainting.h"
#include "Algorithms/CriminisiImageInpainting.h"
#include "Algorithms/FastDigitalImageInpainting.h"
#include "Algorithms/LaplacianImageInpainting.h"
#include "Algorithms/HarmonicImageInpainting.h"

CalculationThread::CalculationThread(MainWindow* mainWindow)
    : mAlgorithmName{""}
    , mAlgorithmType{kNone}
    , mNoise{nullptr}
    , mImageInpaintingBase{nullptr}
    , mMainWindow{mainWindow}
{
}

CalculationThread::~CalculationThread()
{
    DELETE(mNoise);
    DELETE(mImageInpaintingBase);
}

void CalculationThread::initAlgorithm(DataManager* dataManager, ParameterSet* parameterSet)
{
    switch (mAlgorithmType)
    {
        case kNoise:
            mNoise = new Noise(dataManager, parameterSet);
            break;
        case kTeleaImageInpainting:
            mImageInpaintingBase = new TeleaImageInpainting(dataManager, parameterSet);
            break;
        case kNavierStokesImageInpainting:
            mImageInpaintingBase = new NavierStokesImageInpainting(dataManager, parameterSet);
            break;
        case kMaxwellHeavisideImageInpainting:
            mImageInpaintingBase = new MaxwellHeavisideImageInpainting(dataManager, parameterSet);
            break;
        case kBurgersViscousImageInpainting:
            mImageInpaintingBase = new BurgersViscousImageInpainting(dataManager, parameterSet);
            break;
        case kCahnHilliardImageInpainting:
            mImageInpaintingBase = new CahnHilliardImageInpainting(dataManager, parameterSet);
            break;
        case kCriminsiImageInpainting:
            mImageInpaintingBase = new CriminisiImageInpainting(dataManager, parameterSet);
            break;
        case kFastDigitalImageInpainting:
            mImageInpaintingBase = new FastDigitalImageInpainting(dataManager, parameterSet);
            break;
        case kLaplacianImageInpainting:
            mImageInpaintingBase = new LaplacianImageInpainting(dataManager, parameterSet);
        break;
        case kHarmonicImageInpainting:
            mImageInpaintingBase = new HarmonicImageInpainting(dataManager, parameterSet);
            break;
        default:
            break;
    }

    if (mImageInpaintingBase)
    {
        QObject::connect(mImageInpaintingBase, &ImageInpaintingBase::sendImageProcess, mMainWindow, &MainWindow::receiveProcessImage, Qt::QueuedConnection);
        QObject::connect(mImageInpaintingBase, &ImageInpaintingBase::sendOtherMessage, mMainWindow, &MainWindow::receiveOtherMessage, Qt::QueuedConnection);
    }
}

void CalculationThread::run()
{
    emit statusShowMessage("Now applying algorithm --" + mAlgorithmName + "-- ...");
    emit setActionAndWidget(false, false);
    if (mAlgorithmType == kNoise)
        mNoise->addNoise();
    else
        mImageInpaintingBase->inpaint();
    emit setActionAndWidget(true, false);
    emit statusShowMessage("Applying algorithm --" + mAlgorithmName + "-- done.");

    emit needToResetImage();
}
