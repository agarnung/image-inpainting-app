#include "TeleaImageInpainting.h"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

TeleaImageInpainting::TeleaImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void TeleaImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("iters"),
                                5,
                                QString("Iterations number"),
                                QString("The total number of iteration of the algorithm"),
                                true, 1, 100);

    mParameterSet->setName(QString("Telea image inpainting algorithm"));
    mParameterSet->setLabel(QString("Telea image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Parameters settings"));
}

void TeleaImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getOriginalImage();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }

    cv::Mat mask = ~mDataManager->getMask();
    if (mask.empty())
    {
        qWarning() << "The mask is empty";
        return;
    }

    int iters;
    if (!mParameterSet->getValue(QString("iters"), iters))
    {
        qWarning() << "Did not find 'Iterations number'";
        return;
    }

    cv::Mat inpainted;

    if (image.channels() == 4)
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

    if (image.channels() == 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        for (int i = 0; i < (int)channels.size(); ++i)
        {
            emit sendOtherMessage("Processing channel " + QString::number(i) + "...");
            cv::Mat inpaintedChannel;
            cv::inpaint(channels[i], mask, inpaintedChannel, iters, cv::INPAINT_TELEA);
            channels[i] = inpaintedChannel;
        }
        emit sendOtherMessage("");

        cv::merge(channels, inpainted);
    }
    else
    {
        emit sendOtherMessage("Processing image...");
        cv::inpaint(image, mask, inpainted, iters, cv::INPAINT_TELEA);
        emit sendOtherMessage("");
    }

    mDataManager->setImage(image);
    mDataManager->setInpaintedImage(inpainted);

    emit sendImageProcess(inpainted);
}

