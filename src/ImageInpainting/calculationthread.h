#pragma once

#include "mainwindow.h"
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
        CalculationThread(MainWindow* mainWindow = nullptr);
        ~CalculationThread();

        enum AlgorithmsType{kNone, kNoise,
                            kTeleaImageInpainting, kNavierStokesImageInpainting,
                            kMaxwellHeavisideImageInpainting, kBurgersViscousImageInpainting, kCahnHilliardImageInpainting,
                            kCriminsiImageInpainting, kFastDigitalImageInpainting};

        QString mAlgorithmName;
        AlgorithmsType mAlgorithmType;
        Noise* mNoise = nullptr;
        ImageInpaintingBase* mImageInpaintingBase = nullptr;

        void initAlgorithm(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        inline void setAlgorithmName(const QString& algorithmName) { mAlgorithmName = algorithmName; }

        void run();

    private:
        MainWindow* mMainWindow = nullptr;

    signals:
        void needToResetImage();
        void setActionAndWidget(bool value1, bool value2);
        void statusShowMessage(QString message);
};
