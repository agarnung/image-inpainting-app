#include "MaxwellHeavisideImageInpainting.h"

#include <opencv4/opencv2/core.hpp>

MaxwellHeavisideImageInpainting::MaxwellHeavisideImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void MaxwellHeavisideImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("iters"),
                                5,
                                QString("Iterations number"),
                                QString("The total number of iteration of the algorithm"),
                                true, 1, 100);
    mParameterSet->setName(QString("Maxwell-Heaviside image inpainting algorithm"));
    mParameterSet->setLabel(QString("Maxwell-Heaviside image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Maxwell-Heaviside image inpainting algorithm -- Parameters"));
}

void MaxwellHeavisideImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getImage();

    //...
}
