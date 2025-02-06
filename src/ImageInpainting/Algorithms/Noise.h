#pragma once

#include "../parameterset.h"
#include "../datamanager.h"

#include <QList>

class Noise
{
    Q_OBJECT

    public:
        explicit Noise(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~Noise() {}

        void addNoise();

    private:
        enum NoiseType{kRandomMask};

        void initParameters();

        void createRandomMask(double occlusionRatio, cv::Mat& mask);

    private:
        ParameterSet* mParameterSet = nullptr;
        DataManager* mDataManager = nullptr;
};

