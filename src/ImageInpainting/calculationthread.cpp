#include "calculationthread.h"
#include "utils.h"
#include "Algorithms/ImageInpaintingBase.h"
#include "Algorithms/TeleaImageInpainting.h"
#include "Algorithms/NavierStokesImageInpainting.h"
#include "Algorithms/MaxwellHeavisideImageInpainting.h"
#include "Algorithms/CahnHilliardImageInpainting.h"
#include "Algorithms/BurgersViscousImageInpainting.h"

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
        default:
            break;
    }

    if (mImageInpaintingBase)
    {
        QObject::connect(mImageInpaintingBase, &ImageInpaintingBase::sendImageProcess, mMainWindow, &MainWindow::receiveProcessImage);
        QObject::connect(mImageInpaintingBase, &ImageInpaintingBase::sendOtherMessage, mMainWindow, &MainWindow::receiveOtherMessage);
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
