#include "calculationthread.h"
#include "utils.h"
#include "Algorithms/MaxwellHeavisideImageInpainting.h"

CalculationThread::CalculationThread()
    : mAlgorithmName{""}
    , mAlgorithmType{kNone}
    , mNoise{nullptr}
    , mImageInpaintingBase{nullptr}
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
        case kMaxwellHeavisideImageInpainting:
            mImageInpaintingBase = new MaxwellHeavisideInpainting(dataManager, parameterSet);
            break;
        default:
            break;
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

    emit needToUpdate(false);
}

