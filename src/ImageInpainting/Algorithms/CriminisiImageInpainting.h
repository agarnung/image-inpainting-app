#pragma once

#include "ImageInpaintingBase.h"

#include <algorithm>

class Stats;
class Patch;
class TemplateMatchCandidates;
class CriminisiInpainter;

/**
 * @class CriminisiImageInpainting
 * @brief Implementation of the exemplar based inpainting algorithm described in
 *        "Object Removal by Exemplar-Based Inpainting", A. Criminisi et. al.
 *        Based on the implementation by Christoph Heindl (https://github.com/cheind/inpaint, 2014)
 */
class CriminisiImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit CriminisiImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~CriminisiImageInpainting() {}

        void inpaint() override;
        void initParameters() override;

    private:
        /**
         * @param [in, out] image Image to be inpainted. This parameter will be modified with the inpainted result.
         * @param [in] targetMask Region to be inpainted. The mask indicates which parts of the image need to be filled.
         * @param [in] sourceMask Optional mask that specifies the region of the image to synthesize from. If left empty,
         * the entire image (excluding the target mask) will be used.
         * @param [in] patchSize Patch size to use for inpainting.
         */
        void inpaintCriminisi(cv::InputArray image, cv::InputArray targetMask, cv::InputArray sourceMask, int patchSize);
};

class Stats
{
    public:
        template<class T>
        static T clamp(T x, T inclusiveMin, T inclusiveMax)
        {
            return std::clamp(x, inclusiveMin, inclusiveMax);
        }

        template<class T>
        static T clampLower(T x, T inclusiveMin)
        {
            return std::max(x, inclusiveMin);
        }

        template<class T>
        static T clampUpper(T x, T inclusiveMax)
        {
            return std::min(x, inclusiveMax);
        }
};

class Patch
{
    public:
        /** Flags for creating patch. */
        enum PatchFlags {
            /** No flags. Fastest variant. */
            PATCH_FAST = 0,
            /** Clamp patch to bounds of image. */
            PATCH_BOUNDS = 1 << 1,
            /** Reference parent memory. Slower, but keeps the parent memory alive. */
            PATCH_REF = 1 << 2
        };

        /**
                Returns a patch anchored on the given top-left corner.

                \tparam Flags Combination of flags for patch creation.

                \param m Underlying image
                \param y y-coordinate of the patch top-left corner
                \param x x-coordinate of the patch top-left corner
                \param height height of patch (extension along y-axis)
                \param width width of patch (extension along x-axis)
                \return Returns a view on the image that contains only the patch region.
            */
        template<int Flags>
        static cv::Mat topLeftPatch(const cv::Mat &m, int y, int x, int height, int width)
        {
            // Note, compile time if's, will be optimized away by compiler.
            if (Flags & PATCH_BOUNDS)
            {
                int topx = Stats::clamp(x, 0, m.cols - 1);
                int topy = Stats::clamp(y, 0, m.rows - 1);
                width -= std::abs(topx - x);
                height -= std::abs(topy - y);

                width = Stats::clamp(width, 0, m.cols - topx);
                height = Stats::clamp(height, 0, m.rows - topy);
                x = topx;
                y = topy;
            }

            if (Flags & PATCH_REF)
                return m(cv::Rect(x, y, width, height));
            else
            {
                uchar *start = const_cast<uchar*>(m.ptr<uchar>(y, x));
                return cv::Mat(height, width, m.type(), start, m.step);
            }
        }

        /**
                Returns a patch anchored on the given top-left corner..
            */
        static inline cv::Mat topLeftPatch(const cv::Mat &m, int y, int x, int height, int width)
        {
            return topLeftPatch<PATCH_FAST>(m, y, x, height, width);
        }

        /**
                Returns a patch anchored on the given top-left corner..
            */
        static inline cv::Mat topLeftPatch(const cv::Mat &m, const cv::Rect &r)
        {
            return topLeftPatch<PATCH_FAST>(m, r.y, r.x, r.height, r.width);
        }

        /**
                Returns a patch centered around the given pixel coordinates.

                \tparam Flags Combination of flags for patch creation.

                \param m Underlying image
                \param y y-coordinate of the patch center
                \param x x-coordinate of the patch center
                \param halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
                \return Returns a view on the image that contains only the patch region.
            */
        template<int Flags>
        static cv::Mat centeredPatch(const cv::Mat &m, int y, int x, int halfPatchSize)
        {
            int width = 2 * halfPatchSize + 1;
            int height = 2 * halfPatchSize + 1;
            x -= halfPatchSize;
            y -= halfPatchSize;

            return topLeftPatch<Flags>(m, y, x, height, width);
        }

        /**
                Returns a patch centered around the given pixel coordinates.
            */
        static inline cv::Mat centeredPatch(const cv::Mat &m, int y, int x, int halfPatchSize)
        {
            return centeredPatch<PATCH_FAST>(m, y, x, halfPatchSize);
        }

        /**
                Given two centered patches in two images compute the comparable region in both images as top-left patches.

                \param a first image
                \param b second image
                \param ap center in first image
                \param bp center in second image
                \param halfPatchSize halfPatchSize Half the patch size. I.e for a 3x3 patch window, set this to 1.
                \return Comparable rectangles for first, second image. Rectangles are of same size, but anchored top-left
                        with respect to the given center points.
                */
        inline std::pair<cv::Rect, cv::Rect> comparablePatchRegions(
            const cv::Mat &a, const cv::Mat &b,
            cv::Point ap, cv::Point bp,
            int halfPatchSize)
        {
            int left = std::max(-halfPatchSize, std::max(-ap.x, -bp.x));
            int right = std::min(halfPatchSize + 1, std::min(-ap.x + a.cols, -bp.x + b.cols));
            int top = std::max(-halfPatchSize, std::max(-ap.y, -bp.y));
            int bottom = std::min(halfPatchSize + 1, std::min(-ap.y + a.rows, -bp.y + b.rows));

            std::pair<cv::Rect, cv::Rect> p;

            p.first.x = ap.x + left;
            p.first.y = ap.y + top;
            p.first.width = (right - left);
            p.first.height = (bottom - top);

            p.second.x = bp.x + left;
            p.second.y = bp.y + top;
            p.second.width = (right - left);
            p.second.height = (bottom - top);

            return p;
        }

        /** Test if patch goes across the boundary. */
        inline bool isCenteredPatchCrossingBoundary(cv::Point p, int halfPatchSize, const cv::Mat &img)
        {
            return p.x < halfPatchSize || p.x >= img.cols - halfPatchSize ||
                   p.y < halfPatchSize || p.y >= img.rows - halfPatchSize;
        }
};

class TemplateMatchCandidates
{
    public:
        /** Set the source image. */
        void setSourceImage(const cv::Mat &image);

        /** Set the template size. */
        void setTemplateSize(cv::Size templateSize);

        /** Set the partition size. Specifies the number of blocks in x and y direction. */
        void setPartitionSize(cv::Size s);

        /** Initialize candidate search. */
        void initialize();

        /**
            Find candidates.

            \param templ Template image.
            \param templMask Optional template mask.
            \param candidates Computed candidates mask.
            \param maxWeakErrors Max classification mismatches per channel.
            \param maxMeanDifference Max difference of patch / template mean before rejecting a candidate.
            \return Candidate mask.
        */
        void findCandidates(
            const cv::Mat &templ,
            const cv::Mat &templMask,
            cv::Mat &candidates,
            int maxWeakErrors = 3,
            float maxMeanDifference = 20);

        /**
            Find candidate positions for template matching.

            This is a convinience method for using TemplateMatchCandidates.

            \param image Image to search in
            \param templ Template image
            \param templMask Optional template mask
            \param candidate A mask of possible candidates. If image size is W,H and template size is w,h
                   the size of candidate will be W - w + 1, H - h + 1.
            \param partitionSize Number of blocks to subdivide template into
            \param maxWeakErrors Max classification mismatches per channel.
            \param maxMeanDifference Max difference of patch / template mean before rejecting a candidate.
        */
        void findTemplateMatchCandidates(
            cv::InputArray image,
            cv::InputArray templ,
            cv::InputArray templMask,
            cv::OutputArray candidates,
            cv::Size partitionSize = cv::Size(3,3),
            int maxWeakErrors = 3,
            float maxMeanDifference = 20);

    private:

        /** Subdivides a size into a rectangle of blocks. */
        void computeBlockRects(
            cv::Size size,
            cv::Size partitions,
            std::vector< cv::Rect > &rects);

        /** Reject blocks depending on the template mask. */
        void removeInvalidBlocks(
            const cv::Mat &templMask,
            std::vector< cv::Rect > &rects);

        /** Calculate the weak classifiers for the template, taking the mask into account. */
        void weakClassifiersForTemplate(
            const cv::Mat &templ,
            const cv::Mat &templMask,
            const std::vector< cv::Rect > &rects,
            cv::Mat_<int> &classifiers,
            cv::Scalar &mean);

        /** Compare the template classifiers to the classifiers generated from the given template position. */
        uchar compareWeakClassifiers(
            const cv::Mat_<int> &i,
            int x, int y,
            cv::Size templSize,
            const std::vector< cv::Rect > &blocks,
            const int *compareTo,
            float templateMean,
            float maxMeanDiff, int maxWeakErrors);

        cv::Mat mImage;
        std::vector< cv::Mat_<int> > mIntegrals;
        std::vector< cv::Rect > mBlocks;
        cv::Size mTemplateSize;
        cv::Size mPartitionSize;
};

/**
    Implementation of the exemplar based inpainting algorithm described in
    "Object Removal by Exemplar-Based Inpainting", A. Criminisi et. al.

    Changes made by the author (Christoph Heindl) with respect to the original paper:
        - the template match error is calculated based on larger patch sizes than those
          used to infill. The reason behind this is to compare a larger portion of source
          and target regions and thus to avoid visual artefacts.

        - the search for the best matching spot of the patch position to be inpainted
          is accelerated by TemplateMatchCandidates.

    Please note edge cases (i.e regions on the image border) are crudely handled by simply
    discarding them.
*/
class CriminisiInpainter
{
    public:

        /** Empty constructor */
        CriminisiInpainter();

        /** Set the image to be inpainted. */
        void setSourceImage(const cv::Mat& bgrImage);

        /** Set the mask that describes the region inpainting can copy from. */
        void setSourceMask(const cv::Mat& mask);

        /** Set the mask that describes the region to be inpainted. */
        void setTargetMask(const cv::Mat& mask);

        /** Set the patch size. */
        void setPatchSize(int s);

        /** Initialize inpainting. */
        void initialize();

        /** True if there are more steps to perform. */
        bool hasMoreSteps();

        /** Perform a single step (i.e fill one patch) and return the updated information. */
        void step();

        /** Access the current state of the inpainted image. */
        cv::Mat image() const;

        /** Access the current state of the target region. */
        cv::Mat targetRegion() const;
    private:

        /** Updates the fill-front which is the border between filled and unfilled regions. */
        void updateFillFront();

        /** Find patch on fill front with highest priortiy. This will be the patch to be inpainted in this step. */
        cv::Point findTargetPatchLocation();

        /** For a given patch to inpaint, search for the best matching source patch to use for inpainting. */
        cv::Point findSourcePatchLocation(cv::Point targetPatchLocation, bool useCandidateFilter);

        /** Calculate the confidence for the given patch location. */
        float confidenceForPatchLocation(cv::Point p);

        /** Given that we know the source and target patch, propagate associated values from the source into the target region. */
        void propagatePatch(cv::Point target, cv::Point source);

        struct UserSpecified {
            cv::Mat image;
            cv::Mat sourceMask;
            cv::Mat targetMask;
            int patchSize;

            UserSpecified();
        };

        UserSpecified mInput;

        TemplateMatchCandidates mTmc;
        cv::Mat mImage, mCandidates;
        cv::Mat_<uchar> mTargetRegion, mBorderRegion, mSourceRegion;
        cv::Mat_<float> mIsophoteX, mIsophoteY, mConfidence, mBorderGradX, mBorderGradY;
        int mHalfPatchSize, mHalfMatchSize;
        int mStartX, mStartY, mEndX, mEndY;
};

