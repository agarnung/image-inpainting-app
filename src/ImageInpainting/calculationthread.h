#pragma once

#include "datamanager.h"
#include "parameterset.h"
#include "Algorithms/Noise.h"
#include "Algorithms/ImageInpaintingBase.h"

#include <QThread>

/**
 * @class CalculationThread
 * @brief Manages the execution of the inpainting algorithms in a separate thread.
 *
 * This class runs the inpainting algorithms on images in the background.
 */
class CalculationThread: public QThread
{
    Q_OBJECT

    public:
        CalculationThread();
        ~CalculationThread();

        enum AlgorithmsType{kNone, kNoise,
                            kMaxwellHeavisideImageInpainting};

        QString mAlgorithmName;
        AlgorithmsType mAlgorithmType;
        Noise* mNoise = nullptr;
        ImageInpaintingBase* mImageInpaintingBase = nullptr;

        void initAlgorithm(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        inline void setAlgorithmName(const QString& algorithmName) { mAlgorithmName = algorithmName; }

        void run();

    signals:
        void needToResetImage(bool value);
        void setActionAndWidget(bool, bool);
        void statusShowMessage(QString message);
};
