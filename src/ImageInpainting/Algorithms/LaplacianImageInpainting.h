#pragma once

#include "ImageInpaintingBase.h"

#include <opencv4/opencv2/core.hpp>

class LaplacianInpainting;

/**
 * @class LaplacianImageInpainting
 * @brief Implementation of Fast Digital Image Inpainting described in
 *        J. H. Lee, I. Choi and M. H. Kim, "Laplacian Patch-Based Image Synthesis,"
 *        2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
 *        Las Vegas, NV, USA, 2016, pp. 2727-2735, doi: 10.1109/CVPR.2016.298.
 *        (available in http://vclab.kaist.ac.kr/cvpr2016p2/CVPR2016_LaplacianInpainting.pdf)
 * @see   Based on the implementation in https://github.com/KAIST-VCLAB/laplacianinpainting
 */
class LaplacianImageInpainting : public ImageInpaintingBase
{
    Q_OBJECT

    public:
        explicit LaplacianImageInpainting(DataManager* dataManager = nullptr, ParameterSet* parameterSet = nullptr);
        ~LaplacianImageInpainting() {}

        void inpaint() override;
        void initParameters() override;

    private:
        cv::Mat laplacianInpaint(cv::Mat colormat, cv::Mat maskmat, int patchSize = 7, double distanceWeight = 1.3,
                                 double distanceMetric = 0.4, int minSize = 20, int numEM = 50, int decreaseFactor = 10,
                                 int minEMIter = 10, int randomSearchIter = 1);
        void fixDownsampledMaskMat(cv::Mat mask);
        void fixDownsampledMaskMatColorMat(cv::Mat mask,cv::Mat color);
};

class LaplacianInpainting
{
    public:
        int psz_, minsize_;
        double gamma_, highconfidence_, lambda_;
        double siminterval_;
        int patchmatch_iter_;
        int rs_iter_;
        int nnfcount_;
        void findNearestNeighborLap(cv::Mat nnf,cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size,int emiter);
        void colorVoteLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size);
        void doEMIterLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat featuremat, cv::Mat maskmat, std::pair<int, int> size,int num_emiter, cv::Size orig_size, char *processfilename);
        void constructLaplacianPyr(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat &img);
        void constructLaplacianPyrMask(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat mask,cv::Mat &img);
        void upscaleImages(cv::Mat nnf, cv::Mat nnferr, bool *patch_type,  cv::Mat colorfmat,  cv::Mat dmaskmat,  cv::Mat umaskmat);

        __inline double computePatchError(double *patcha, double *patchb, int psz){
            int pixeln = psz*psz*3; // 3 channels
            double sum = 0;
            for(int i = 0 ; i < pixeln; i++)
                sum+=(patcha[i]-patchb[i])*(patcha[i]-patchb[i]);
            return sum;
        }

        __inline double computePatchErrorLap(std::vector<double*> &colorpatches, std::vector<double*> &colorfpatches,int x, int y, int psz, double lambda){
            int pixeln = psz*psz*3; // 3 channels

            double *patcha, *patchb, *patchfa, *patchfb;

            patcha = (double*)colorpatches[x];
            patchb = (double*)colorpatches[y];
            patchfa = (double*)colorfpatches[x];
            patchfb = (double*)colorfpatches[y];

            double sum = 0;
            for(int i = 0 ; i < pixeln; i++)
                sum += (1-lambda)*(patcha[i]-patchb[i])*(patcha[i]-patchb[i]);
            for(int i = 0 ; i < pixeln; i++)
                sum += (lambda)*(patchfa[i]-patchfb[i])*(patchfa[i]-patchfb[i]);
            return sum;
        }
};
