#include "FastDigitalImageInpainting.h"

#include "utils.h"

FastDigitalImageInpainting::FastDigitalImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void FastDigitalImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("maxIters"),
                                500,
                                QString("maxIters"),
                                QString("maxIters"),
                                true, 1, 9999);
    mParameterSet->addParameter(QString("a"),
                                0.073235f,
                                QString("a"),
                                QString("a"),
                                true, 0.000001f, 1.0f);
    mParameterSet->addParameter(QString("b"),
                                0.176765f,
                                QString("b"),
                                QString("b"),
                                true, 0.000001f, 1.0f);

    mParameterSet->setName(QString("Fast Digital Image Inpainting algorithm"));
    mParameterSet->setLabel(QString("Fast Digital Image Inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Parameters settings"));
}

void FastDigitalImageInpainting::fastInpaint(const cv::Mat& src, const cv::Mat& mask, const cv::Mat& kernel, cv::Mat& dst, int maxNumOfIter)
{
    assert(src.type() == mask.type() && mask.type() == CV_8UC3);
    assert(src.size() == mask.size());
    assert(kernel.type() == CV_32F);

    // fill in the missing region with the input's average color
    auto avgColor = cv::sum(src) / (src.cols * src.rows);
    cv::Mat avgColorMat(1, 1, CV_8UC3);
    avgColorMat.at<cv::Vec3b>(0, 0) = cv::Vec3b(avgColor[0], avgColor[1], avgColor[2]);
    cv::resize(avgColorMat, avgColorMat, src.size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat filledRegion = (mask / 255).mul(src) + (1 - mask / 255).mul(avgColorMat);

    // Aplica difusión anisotrópica para preservar los bordes
    //    cv::ximgproc::anisotropicDiffusion(filledRegion, dst, 0.15, 0.1, 3);
    //    cv::imshow("diffused", dst); cv::waitKey(0);

    filledRegion.copyTo(dst);

    // convolution
    int bSize = kernel.cols / 2;
    cv::Mat kernel3ch, inWithBorder;
    dst.convertTo(dst, CV_32FC3);
    cv::cvtColor(kernel, kernel3ch, cv::COLOR_GRAY2BGR);

    cv::copyMakeBorder(dst, inWithBorder, bSize, bSize, bSize, bSize, cv::BORDER_REPLICATE);
    cv::Mat resInWithBorder = cv::Mat(inWithBorder, cv::Rect(bSize, bSize, dst.cols, dst.rows));

    const int ch = dst.channels();
    for (int itr = 0; itr < maxNumOfIter; ++itr)
    {
        cv::copyMakeBorder(dst, inWithBorder, bSize, bSize, bSize, bSize, cv::BORDER_REPLICATE);

        for (int r = 0; r < dst.rows; ++r)
        {
            const uchar* pMask = mask.ptr(r);
            float* pRes = dst.ptr<float>(r);
            for (int c = 0; c < dst.cols; ++c)
            {
                if (pMask[ch * c] == 0)
                {
                    cv::Rect rectRoi(c, r, kernel.cols, kernel.rows);
                    cv::Mat roi(inWithBorder, rectRoi);

                    auto sum = cv::sum(kernel3ch.mul(roi));
                    pRes[ch * c + 0] = sum[0];
                    pRes[ch * c + 1] = sum[1];
                    pRes[ch * c + 2] = sum[2];
                }
            }
        }

        cv::Mat sentImage;
        dst.convertTo(sentImage, CV_8UC3);
        emit sendImageProcess(sentImage);
    }

    dst.convertTo(dst, CV_8UC3);
}

void FastDigitalImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getOriginalImage();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }
    image = universalConvertTo(image, CV_8UC3);

    cv::Mat mask = mDataManager->getMask();
    if (mask.empty())
    {
        qWarning() << "The mask is empty";
        return;
    }
    mask = universalConvertTo(mask, CV_8UC3);

    int maxIters;
    double a, b;
    if (!mParameterSet->getValue(QString("a"), a) ||
        !mParameterSet->getValue(QString("b"), b) ||
        !mParameterSet->getValue(QString("maxIters"), maxIters))
    {
        qWarning() << "Could not retrieve all parameters.";
        return;
    }
    const float a_float = (float)a;
    const float b_float = (float)b;
    const cv::Mat kernel = (cv::Mat_<float>(3, 3) << a_float, b_float, a_float, b_float, 0.0f, b_float, a_float, b_float, a_float);

    cv::Mat inpainted;
    emit sendOtherMessage("Processing image...");
    fastInpaint(image, mask, kernel, inpainted, maxIters);
    emit sendOtherMessage("");

    mDataManager->setImage(image);
    mDataManager->setInpaintedImage(inpainted);

    emit sendImageProcess(inpainted);
}
