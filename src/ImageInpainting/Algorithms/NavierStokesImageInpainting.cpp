#include "NavierStokesImageInpainting.h"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

NavierStokesImageInpainting::NavierStokesImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void NavierStokesImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("iters"),
                                5,
                                QString("Iterations number"),
                                QString("The total number of iteration of the algorithm"),
                                true, 1, 100);

    mParameterSet->setName(QString("Navier-Stokes image inpainting algorithm"));
    mParameterSet->setLabel(QString("Navier-Stokes image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Navier-Stokes image inpainting algorithm -- Parameters"));
}

void NavierStokesImageInpainting::inpaint()
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
            cv::Mat inpaintedChannel;
            cv::inpaint(channels[i], mask, inpaintedChannel, iters, cv::INPAINT_NS);
            channels[i] = inpaintedChannel;
        }

        cv::merge(channels, inpainted);
    }
    else
        cv::inpaint(image, mask, inpainted, iters, cv::INPAINT_NS);

    mDataManager->setInpaintedImage(inpainted);
}

