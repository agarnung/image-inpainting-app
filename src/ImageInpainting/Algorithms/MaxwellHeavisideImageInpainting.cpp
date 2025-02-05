#include "MaxwellHeavisideImageInpainting.h"

#include <opencv4/opencv2/core.hpp>

MaxwellHeavisideInpainting::MaxwellHeavisideInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void MaxwellHeavisideInpainting::initParameters()
{
}

void MaxwellHeavisideInpainting::inpaint()
{
    cv::Mat image = mDataManager->getImage();

    //...
}
