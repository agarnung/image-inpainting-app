#pragma once

#include "../parameterset.h"
#include "../datamanager.h"

class ImageInpaintingBase : public QObject
{
    Q_OBJECT

    public:
        explicit ImageInpaintingBase(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        virtual ~ImageInpaintingBase() {}

    public:
        virtual void inpaint() = 0;
        virtual void initParameters() = 0;

    protected:
        ParameterSet* mParameterSet = nullptr;
        DataManager* mDataManager = nullptr;

    signals:
        void sendImageProcess(const cv::Mat& img);
        void sendOtherMessage(const QString& msg);
};

