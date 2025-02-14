#include "Noise.h"

#include <QDebug>

Noise::Noise(DataManager* dataManager, ParameterSet* parameterSet)
{
    if ((dataManager == nullptr) || (parameterSet == nullptr))
        return;

    mParameterSet = parameterSet;
    mDataManager = dataManager;

    initParameters();
}

void Noise::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("Occlusion ratio"),
                                0.75,
                                QString("Occlusion ratio"),
                                QString("The desired ratio of lost by original observed pixels"),
                                true, 0.0, 1.0);

    QStringList strListNoiseType;
    strListNoiseType.push_back(QString("Random mask"));
    mParameterSet->addParameter(QString("Noise type"),
                                strListNoiseType,
                                0,
                                QString("Noise type"),
                                QString("The type of noise"));

    mParameterSet->setName(QString("Noise"));
    mParameterSet->setLabel(QString("Noise"));
    mParameterSet->setIntroduction(QString("Noise -- Parameters"));
}

void Noise::createRandomMask(double occlusionRatio, cv::Mat& mask)
{
    mask.setTo(cv::Scalar(1));
    occlusionRatio = std::clamp(occlusionRatio, 0.0, 1.0);

    if (mask.type() == CV_32FC1 || mask.type() == CV_64FC1)
    {
        cv::randu(mask, cv::Scalar(0), cv::Scalar(1));

        for (int y = 0; y < mask.rows; ++y)
        {
            auto* p_mask = mask.ptr<double>(y);
            for (int x = 0; x < mask.cols; ++x)
                p_mask[x] = (p_mask[x] < occlusionRatio) ? 0.0 : 1.0;
        }
    }
    else
    {
        cv::Mat tempMask(mask.size(), CV_64FC1);
        cv::randu(tempMask, cv::Scalar(0.0), cv::Scalar(1.0));

        for (int y = 0; y < tempMask.rows; ++y)
        {
            double* p_mask = tempMask.ptr<double>(y);
            for (int x = 0; x < tempMask.cols; ++x)
                p_mask[x] = (p_mask[x] < occlusionRatio) ? 0.0 : 1.0;
        }

        tempMask.convertTo(mask, mask.type());
    }
}

void Noise::addNoise()
{
    cv::Mat image = mDataManager->getOriginalImage().clone();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }

    double occlusionRatio;
    int noiseTypeIndex;
    if (!mParameterSet->getValue(QString("Occlusion ratio"), occlusionRatio))
    {
        qWarning() << "Did not find 'Occlusion ratio'";
        return;
    }
    if (!mParameterSet->getStringListIndex(QString("Noise type"), noiseTypeIndex))
    {
        qWarning() << "Did not find 'Noise type'";
        return;
    }

    NoiseType noiseType = static_cast<NoiseType>(noiseTypeIndex);

    switch (noiseType)
    {
        case kRandomMask:
        {
            int originalType = image.type();

            cv::Mat imageConverted;
            if (image.channels() == 1)
                image.convertTo(imageConverted, CV_64FC1);
            else
            {
                std::vector<cv::Mat> channels;
                cv::split(image, channels);
                for (auto& channel : channels)
                    channel.convertTo(channel, CV_64FC1);
                cv::merge(channels, imageConverted);
            }

            cv::Mat mask(image.size(), CV_64FC1);
            createRandomMask(occlusionRatio, mask);

            if (imageConverted.channels() == 1)
                cv::multiply(imageConverted, mask, imageConverted);
            else
            {
                std::vector<cv::Mat> channels;
                cv::split(imageConverted, channels);
                for (auto& channel : channels)
                    cv::multiply(channel, mask, channel);
                cv::merge(channels, imageConverted);
            }

            imageConverted.convertTo(image, originalType);

            mask.convertTo(mask, CV_8UC1, 255.0, 0.0);
            mDataManager->setMask(mask);

            break;
        }

        default:
            break;
    }

    mDataManager->setImage(image.clone());
    mDataManager->setNoisyImage(image.clone());
    mDataManager->setInpaintedImage(image.clone());
}
