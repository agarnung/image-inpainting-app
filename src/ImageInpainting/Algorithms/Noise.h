#pragma once

#include "../parameterset.h"
#include "../datamanager.h"

#include <QList>
#include <QObject>

class Noise
{
    public:
        explicit Noise(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~Noise() {}

        void addNoise();

    private:
        enum NoiseType{kRandomMask, kRandomSquares, kRandomShapes};

        void initParameters();

        void createRandomMask(double occlusionRatio, cv::Mat& mask);
        void createRandomRectangles(cv::Mat& image, int numRectangles, int rectWidth, int rectHeight, bool drawWites = true);

    private:
        ParameterSet* mParameterSet = nullptr;
        DataManager* mDataManager = nullptr;
};

