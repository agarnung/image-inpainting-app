#include "CriminisiImageInpainting.h"

#include <opencv4/opencv2/core.hpp>

#include <QDebug>

CriminisiImageInpainting::CriminisiImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void CriminisiImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("patchSize"),
                                5,
                                QString("patchSize"),
                                QString("Patch size"),
                                true, 2, 200);

    mParameterSet->setName(QString("Criminisi image inpainting algorithm"));
    mParameterSet->setLabel(QString("Criminisi image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Criminisi settings"));
}

void CriminisiImageInpainting::inpaintCriminisi(cv::InputArray image, cv::InputArray targetMask, cv::InputArray sourceMask, int patchSize)
{
    CriminisiInpainter ci;
    ci.setSourceImage(image.getMat());
    ci.setSourceMask(sourceMask.getMat());
    ci.setTargetMask(targetMask.getMat());
    ci.setPatchSize(patchSize);
    ci.initialize();

    while (ci.hasMoreSteps())
    {
        ci.step();
        emit sendImageProcess(ci.image());
    }

    ci.image().copyTo(image.getMat());
}

void CriminisiImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getOriginalImage();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }

    cv::Mat targetMask = ~mDataManager->getMask();
    if (targetMask.empty())
    {
        qWarning() << "The mask is empty";
        return;
    }

    int patchSize;
    if (!mParameterSet->getValue(QString("patchSize"), patchSize))
    {
        qWarning() << "Did not find 'patchSize'";
        return;
    }

    cv::Mat inpainted = image.clone();

    if (inpainted.channels() == 4)
        cv::cvtColor(inpainted, inpainted, cv::COLOR_BGRA2BGR);

    if (inpainted.channels() == 1)
        cv::cvtColor(inpainted, inpainted, cv::COLOR_GRAY2BGR);

    CriminisiInpainter inpainter;
    cv::Mat sourceMask;
    sourceMask.create(inpainted.size(), CV_8UC1);
    sourceMask.setTo(0);
    emit sendOtherMessage("Processing image...");
    inpaintCriminisi(inpainted, targetMask, sourceMask, patchSize);
    emit sendOtherMessage("");

    mDataManager->setImage(image);
    mDataManager->setInpaintedImage(inpainted);

    emit sendImageProcess(inpainted);
}

const int PATCHFLAGS = Patch::PATCH_BOUNDS;

CriminisiInpainter::UserSpecified::UserSpecified()
{
    patchSize = 9;
}

CriminisiInpainter::CriminisiInpainter()
{
    ;
}

void CriminisiInpainter::setSourceImage(const cv::Mat &bgrImage)
{
    mInput.image = bgrImage;
}

void CriminisiInpainter::setTargetMask(const cv::Mat &mask)
{
    mInput.targetMask = mask;
}

void CriminisiInpainter::setSourceMask(const cv::Mat &mask)
{
    mInput.sourceMask = mask;
}

void CriminisiInpainter::setPatchSize(int s)
{
    mInput.patchSize = s;
}

cv::Mat CriminisiInpainter::image() const
{
    return mImage;
}

cv::Mat CriminisiInpainter::targetRegion() const
{
    return mTargetRegion;
}

void CriminisiInpainter::initialize()
{
    CV_Assert(mInput.image.channels() == 3);
    CV_Assert(mInput.image.depth() == CV_8U);
    CV_Assert( mInput.targetMask.size() == mInput.image.size());
    CV_Assert(mInput.sourceMask.empty() || mInput.targetMask.size() == mInput.sourceMask.size());
    CV_Assert(mInput.patchSize > 0);

    mHalfPatchSize = mInput.patchSize / 2;
    mHalfMatchSize = (int) (mHalfPatchSize * 1.25f);

    mInput.image.copyTo(mImage);
    mInput.targetMask.copyTo(mTargetRegion);

    // Initialize regions
    cv::rectangle(mTargetRegion, cv::Rect(0, 0, mTargetRegion.cols, mHalfMatchSize), cv::Scalar(0), -1); // Top
    cv::rectangle(mTargetRegion, cv::Rect(0, 0, mHalfMatchSize, mTargetRegion.rows), cv::Scalar(0), -1); // Left
    cv::rectangle(mTargetRegion, cv::Rect(mTargetRegion.cols - mHalfMatchSize, 0, mTargetRegion.cols, mTargetRegion.rows), cv::Scalar(0), -1); // Bottom
    cv::rectangle(mTargetRegion, cv::Rect(0, mTargetRegion.rows - mHalfMatchSize, mTargetRegion.cols, mTargetRegion.rows), cv::Scalar(0), -1); // Right

    mSourceRegion = 255 - mTargetRegion;
    cv::rectangle(mSourceRegion, cv::Rect(0, 0, mSourceRegion.cols, mSourceRegion.rows), cv::Scalar(0), mHalfMatchSize);
    cv::erode(mSourceRegion, mSourceRegion, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(mHalfMatchSize*2+1, mHalfMatchSize*2+1)));

    if (!mInput.sourceMask.empty() && cv::countNonZero(mInput.sourceMask) > 0)
        mSourceRegion.setTo(0, (mInput.sourceMask == 0));

    // Initialize isophote values. Deviating from the original paper here. We've found that
    // blurring the image balances the data term and the confidence term better.
    cv::Mat blurred;
    cv::blur(mImage, blurred, cv::Size(3,3));
    cv::Mat_<cv::Vec3f> gradX, gradY;
    cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

    mIsophoteX.create(gradX.size());
    mIsophoteY.create(gradY.size());

    for (int i = 0; i < gradX.rows * gradX.cols; ++i)
    {
        // Note the isophote corresponds to the gradient rotated by 90 degrees
        const cv::Vec3f& vx = gradX(i);
        const cv::Vec3f& vy = gradY(i);

        float x = (vx[0] + vx[1] + vx[2]) / (3 * 255);
        float y = (vy[0] + vy[1] + vy[2]) / (3 * 255);

        std::swap(x, y);
        x *= -1;

        mIsophoteX(i) = x;
        mIsophoteY(i) = y;
    }

    // Initialize confidence values
    mConfidence.create(mImage.size());
    mConfidence.setTo(1);
    mConfidence.setTo(0, mTargetRegion);

    // Configure valid image region considered during algorithm
    mStartX = mHalfMatchSize;
    mStartY = mHalfMatchSize;
    mEndX = mImage.cols - mHalfMatchSize - 1;
    mEndY = mImage.rows - mHalfMatchSize - 1;

    // Setup template match performance improvement
    mTmc.setSourceImage(mImage);
    mTmc.setTemplateSize(cv::Size(mHalfMatchSize * 2 + 1, mHalfMatchSize * 2 + 1));
    mTmc.setPartitionSize(cv::Size(3,3));
    mTmc.initialize();
}

bool CriminisiInpainter::hasMoreSteps()
{
    return cv::countNonZero(mTargetRegion) > 0;
}

void CriminisiInpainter::step()
{
    // We also need an updated knowledge of gradients in the border region
    updateFillFront();

    // Next, we need to select the best target patch on the boundary to be inpainted.
    cv::Point targetPatchLocation = findTargetPatchLocation();

    // Determine the best matching source patch from which to inpaint.
    cv::Point sourcePatchLocation = findSourcePatchLocation(targetPatchLocation, true);
    if (sourcePatchLocation.x == -1)
        sourcePatchLocation = findSourcePatchLocation(targetPatchLocation, false);

    // Copy values
    propagatePatch(targetPatchLocation, sourcePatchLocation);
}

void CriminisiInpainter::updateFillFront()
{
    // 2nd order derivative used to find border.
    cv::Laplacian(mTargetRegion, mBorderRegion, CV_8U, 3, 1, 0, cv::BORDER_REPLICATE);

    // Update confidence values along fill front.
    for (int y = mStartY; y < mEndY; ++y)
    {
        const uchar *bRow = mBorderRegion.ptr(y);
        for (int x = mStartX; x < mEndX; ++x)
        {
            if (bRow[x] > 0)
            {
                // Update confidence for border item
                cv::Point p(x, y);
                mConfidence(p) = confidenceForPatchLocation(p);
            }
        }
    }
}

cv::Point CriminisiInpainter::findTargetPatchLocation()
{
    // Sweep over all pixels in the border region and priorize them based on
    // a confidence term (i.e how many pixels are already known) and a data term that prefers
    // border pixels on strong edges running through them.

    float maxPriority = 0;
    cv::Point bestLocation(0, 0);

    mBorderGradX.create(mTargetRegion.size());
    mBorderGradY.create(mTargetRegion.size());
    cv::Sobel(mTargetRegion, mBorderGradX, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(mTargetRegion, mBorderGradY, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

    for (int y = mStartY; y < mEndY; ++y)
    {
        const uchar* bRow = mBorderRegion.ptr(y);
        const float* gxRow = mBorderGradX.ptr<float>(y);
        const float* gyRow = mBorderGradY.ptr<float>(y);
        const float* ixRow = mIsophoteX.ptr<float>(y);
        const float* iyRow = mIsophoteY.ptr<float>(y);
        const float* cRow = mConfidence.ptr<float>(y);

        for (int x = mStartX; x < mEndX; ++x)
        {
            if (bRow[x] > 0)
            {

                // Data term
                cv::Vec2f grad(gxRow[x], gyRow[x]);
                float dot = grad.dot(grad);

                if (dot == 0)
                    grad *= 0;
                else
                    grad /= sqrtf(dot);

                const float d = fabs(grad[0] * ixRow[x] + grad[1] * iyRow[x]) + 0.0001f;

                // Confidence term
                const float c = cRow[x];

                // Priority of patch
                const float prio = c * d;

                if (prio > maxPriority)
                {
                    maxPriority = prio;
                    bestLocation = cv::Point(x,y);
                }
            }
        }
    }

    return bestLocation;
}

float CriminisiInpainter::confidenceForPatchLocation(cv::Point p)
{
    cv::Mat_<float> c = Patch::centeredPatch<PATCHFLAGS>(mConfidence, p.y, p.x, mHalfPatchSize);
    return (float)cv::sum(c)[0] / c.size().area();
}

cv::Point CriminisiInpainter::findSourcePatchLocation(cv::Point targetPatchLocation, bool useCandidateFilter)
{
    cv::Point bestLocation(-1, -1);
    float bestError = std::numeric_limits<float>::max();

    cv::Mat_<cv::Vec3b> targetImagePatch = Patch::centeredPatch<PATCHFLAGS>(mImage, targetPatchLocation.y, targetPatchLocation.x, mHalfMatchSize);
    cv::Mat_<uchar> targetMask = Patch::centeredPatch<PATCHFLAGS>(mTargetRegion, targetPatchLocation.y, targetPatchLocation.x, mHalfMatchSize);

    cv::Mat invTargetMask = (targetMask == 0);
    if (useCandidateFilter)
        mTmc.findCandidates(targetImagePatch, invTargetMask, mCandidates, 3, 10);

    for (int y = mStartY; y < mEndY; ++y)
    {
        for (int x = mStartX; x < mEndX; ++x)
        {

            // Note, candidates need to be corrected. Centered patch locations used here, top-left used with candidates.
            const bool shouldTest = (!useCandidateFilter || mCandidates.at<uchar>(y - mHalfMatchSize, x - mHalfMatchSize)) &&
                                    mSourceRegion.at<uchar>(y, x) > 0;

            if (shouldTest)
            {
                cv::Mat_<uchar> sourceMask = Patch::centeredPatch<PATCHFLAGS>(mSourceRegion, y, x, mHalfMatchSize);
                cv::Mat_<cv::Vec3b> sourceImagePatch = Patch::centeredPatch<PATCHFLAGS>(mImage, y, x, mHalfMatchSize);

                float error = (float)cv::norm(targetImagePatch, sourceImagePatch, cv::NORM_L1, invTargetMask);

                if (error < bestError)
                {
                    bestError = error;
                    bestLocation = cv::Point(x, y);
                }
            }
        }
    }

    return bestLocation;
}

void CriminisiInpainter::propagatePatch(cv::Point target, cv::Point source)
{
    cv::Mat_<uchar> copyMask = Patch::centeredPatch<PATCHFLAGS>(mTargetRegion, target.y, target.x, mHalfPatchSize);

    Patch::centeredPatch<PATCHFLAGS>(mImage, source.y, source.x, mHalfPatchSize).copyTo(
        Patch::centeredPatch<PATCHFLAGS>(mImage, target.y, target.x, mHalfPatchSize),
        copyMask);

    Patch::centeredPatch<PATCHFLAGS>(mIsophoteX, source.y, source.x, mHalfPatchSize).copyTo(
        Patch::centeredPatch<PATCHFLAGS>(mIsophoteX, target.y, target.x, mHalfPatchSize),
        copyMask);

    Patch::centeredPatch<PATCHFLAGS>(mIsophoteY, source.y, source.x, mHalfPatchSize).copyTo(
        Patch::centeredPatch<PATCHFLAGS>(mIsophoteY, target.y, target.x, mHalfPatchSize),
        copyMask);

    float cPatch = mConfidence.at<float>(target);
    Patch::centeredPatch<PATCHFLAGS>(mConfidence, target.y, target.x, mHalfPatchSize).setTo(cPatch, copyMask);

    copyMask.setTo(0);
}

void TemplateMatchCandidates::setSourceImage(const cv::Mat &image)
{
    CV_Assert(image.channels() == 1 || image.channels() == 3);
    CV_Assert(image.depth() == CV_8U);

    mImage = image;
}

void TemplateMatchCandidates::setTemplateSize(cv::Size templateSize)
{
    mTemplateSize = templateSize;
}

void TemplateMatchCandidates::setPartitionSize(cv::Size partitionSize)
{
    mPartitionSize = partitionSize;
}

void TemplateMatchCandidates::initialize()
{
    std::vector<cv::Mat_<uchar>> imageChannels;
    cv::split(mImage, imageChannels);
    const size_t nChannels = imageChannels.size();

    mIntegrals.resize(nChannels);
    for (size_t i = 0; i < nChannels; ++i)
        cv::integral(imageChannels[i], mIntegrals[i]);

    mBlocks.clear();
    computeBlockRects(mTemplateSize, mPartitionSize, mBlocks);
}

void TemplateMatchCandidates::findCandidates(
    const cv::Mat& templ,
    const cv::Mat& templMask,
    cv::Mat &candidates,
    int maxWeakErrors,
    float maxMeanDifference)
{
    CV_Assert(templ.type() == (int)CV_MAKETYPE(CV_8U, mIntegrals.size()));
    CV_Assert(templ.size() == mTemplateSize);
    CV_Assert(templMask.empty() || templMask.size() == mTemplateSize);

    candidates.create(
        mImage.size().height - templ.size().height + 1,
        mImage.size().width - templ.size().width + 1,
        CV_8UC1);
    candidates.setTo(255);

    std::vector< cv::Rect > blocks = mBlocks;
    removeInvalidBlocks(templMask, blocks);

    cv::Mat_<int> referenceClass;
    cv::Scalar templMean;
    weakClassifiersForTemplate(templ, templMask, blocks, referenceClass, templMean);

    // For each channel we loop over all possible template positions and compare with classifiers.
    for (size_t i = 0; i < mIntegrals.size(); ++i)
    {
        cv::Mat_<int> &integral = mIntegrals[i];
        const int *referenceClassRow = referenceClass.ptr<int>(static_cast<int>(i));

        // For all template positions ty, tx (top-left template position)
        for (int ty = 0; ty < candidates.rows; ++ty)
        {
            uchar *outputRow = candidates.ptr<uchar>(ty);

            for (int tx = 0; tx < candidates.cols; ++tx)
            {
                if (!outputRow[tx])
                    continue;

                outputRow[tx] = compareWeakClassifiers(
                    integral,
                    tx, ty,
                    templ.size(),
                    blocks,
                    referenceClassRow,
                    (float)templMean[static_cast<int>(i)],
                    maxMeanDifference,
                    maxWeakErrors);
            }
        }
    }
}

void TemplateMatchCandidates::weakClassifiersForTemplate(
    const cv::Mat &templ,
    const cv::Mat &templMask,
    const std::vector< cv::Rect > &rects,
    cv::Mat_<int> &classifiers,
    cv::Scalar &mean)
{
    const int nChannels = templ.channels();
    classifiers.create(nChannels, (int)rects.size());

    // Note we use cv::mean here to make use of mask.
    mean = cv::mean(templ, templMask);

    for (int x = 0; x < (int)rects.size(); ++x)
    {
        cv::Scalar blockMean = cv::mean(templ(rects[x]), templMask.empty() ? cv::noArray() : templMask(rects[x]));

        for (int y = 0; y < nChannels; ++y)
            classifiers(y, x) = blockMean[y] > mean[y] ? 1 : -1;
    }
}

uchar TemplateMatchCandidates::compareWeakClassifiers(
    const cv::Mat_<int> &i,
    int x, int y,
    cv::Size templSize,
    const std::vector< cv::Rect > &blocks,
    const int *compareTo,
    float templateMean,
    float maxMeanDiff, int maxWeakErrors)
{
    const int *topRow = i.ptr<int>(y);
    const int *bottomRow = i.ptr<int>(y + templSize.height); // +1 required for integrals

    // Mean of image under given template position
    const float posMean = (bottomRow[x + templSize.width] - bottomRow[x] - topRow[x + templSize.width] + topRow[x]) / (1.f * templSize.area());

    if  (std::abs(posMean - templateMean) > maxMeanDiff)
        return 0;

    // Evaluate means of sub-blocks
    int sumErrors = 0;
    for (size_t r = 0; r < blocks.size(); ++r)
    {
        const cv::Rect &b = blocks[r];

        int ox = x + b.x;
        int oy = y + b.y;

        const int *topRow = i.ptr<int>(oy);
        const int *bottomRow = i.ptr<int>(oy + b.height);

        const float blockMean = (bottomRow[ox + b.width] - bottomRow[ox] - topRow[ox + b.width] + topRow[ox]) / (1.f * b.width * b.height);
        const int c = blockMean > posMean ? 1 : -1;
        sumErrors += (c != compareTo[r]) ? 1 : 0;

        if (sumErrors > maxWeakErrors)
            return 0;
    }

    return 255;
}

void TemplateMatchCandidates::computeBlockRects(cv::Size size, cv::Size partitions, std::vector< cv::Rect > &rects)
{
    rects.clear();

    const int blockWidth = size.width / partitions.width;
    const int blockHeight = size.height / partitions.height;


    if (blockWidth == 0 || blockHeight == 0)
        rects.push_back(cv::Rect(0, 0, size.width, size.height));
    else
    {
        // Note: last row/column of blocks might be of different shape to fill up entire size.
        const int lastBlockWidth = size.width - blockWidth * (partitions.width - 1);
        const int lastBlockHeight = size.height - blockHeight * (partitions.height - 1);

        for (int y = 0; y < partitions.height; ++y)
        {
            bool lastY = (y == partitions.height - 1);
            for (int x = 0; x < partitions.width; ++x)
            {
                bool lastX = (x == partitions.width - 1);

                rects.push_back(cv::Rect(
                    x * blockWidth,
                    y * blockHeight,
                    lastX ? lastBlockWidth : blockWidth,
                    lastY ? lastBlockHeight : blockHeight));
            }
        }
    }
}

void TemplateMatchCandidates::removeInvalidBlocks(const cv::Mat &templMask, std::vector< cv::Rect > &rects)
{
    if (!templMask.empty())
    {
        rects.erase(std::remove_if(rects.begin(), rects.end(), [&templMask](const cv::Rect &r) -> bool {
                        cv::Mat block = templMask(r);
                        return cv::countNonZero(block) != block.size().area();
                    }), rects.end());
    }
}

void TemplateMatchCandidates::findTemplateMatchCandidates(
    cv::InputArray image,
    cv::InputArray templ,
    cv::InputArray templMask,
    cv::OutputArray candidates,
    cv::Size partitionSize,
    int maxWeakErrors,
    float maxMeanDifference)
{
    TemplateMatchCandidates tmc;
    tmc.setSourceImage(image.getMat());
    tmc.setPartitionSize(partitionSize);
    tmc.setTemplateSize(templ.size());
    tmc.initialize();

    candidates.create(
        image.size().height - templ.size().height + 1,
        image.size().width - templ.size().width + 1,
        CV_8UC1);

    tmc.findCandidates(templ.getMat(), templMask.getMat(), candidates.getMatRef(), maxWeakErrors, maxMeanDifference);
}
