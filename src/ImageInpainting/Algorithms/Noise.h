#pragma once

#include "../parameterset.h"
#include "../datamanager.h"

#include <vector>
#include <utility>
#include <algorithm>
#include <QList>

class Noise
{
    public:
        explicit Noise(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~Noise() {}

    public:
        void addNoise();

    private:
        void initParameters();

    private:
        ParameterSet* mParameterSet = nullptr;
        DataManager* mDataManager = nullptr;
};

