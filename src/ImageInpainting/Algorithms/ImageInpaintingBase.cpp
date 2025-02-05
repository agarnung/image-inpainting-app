#include "ImageInpaintingBase.h"

ImageInpaintingBase::ImageInpaintingBase(DataManager* dataManager, ParameterSet* parameterSet)
{
    if((dataManager == nullptr) || (parameterSet == nullptr))
        return;

    mParameterSet = parameterSet;
    mDataManager = dataManager;
}

