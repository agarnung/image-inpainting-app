#include "Noise.h"

Noise::Noise(DataManager* dataManager, ParameterSet* parameterSet)
{
    if((dataManager == nullptr) || (parameterSet == nullptr))
        return;

    mParameterSet = parameterSet;
    mDataManager = dataManager;

    initParameters();
}

void Noise::initParameters()
{
    ;
}

void Noise::addNoise()
{
    ;
}

