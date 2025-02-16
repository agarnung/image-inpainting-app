#include "LaplacianImageInpainting.h"

#include "utils.h"

#include <QDebug>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/photo.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

LaplacianImageInpainting::LaplacianImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void LaplacianImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("patchSize"),
                                7,
                                QString("patchSize"),
                                QString("Size of the patches to use"),
                                true, 1, 100);

    mParameterSet->addParameter(QString("distanceWeight"),
                                1.3,
                                QString("distanceWeight"),
                                QString("Weight for the distance function"),
                                true, 0.1, 10.0);

    mParameterSet->addParameter(QString("distanceMetric"),
                                0.4,
                                QString("distanceMetric"),
                                QString("Distance metric for the inpainting process"),
                                true, 0.0, 1.0);

    mParameterSet->addParameter(QString("minSize"),
                                20,
                                QString("minSize"),
                                QString("Minimum size of patches considered for inpainting"),
                                true, 1, 100);

    mParameterSet->addParameter(QString("numEM"),
                                50,
                                QString("numEM"),
                                QString("Number of Expectation-Maximization iterations"),
                                true, 1, 1000);

    mParameterSet->addParameter(QString("decreaseFactor"),
                                10,
                                QString("decreaseFactor"),
                                QString("Factor by which values decrease during optimization"),
                                true, 1, 100);

    mParameterSet->addParameter(QString("minEMIter"),
                                10,
                                QString("minEMIter"),
                                QString("Minimum number of iterations for Expectation-Maximization"),
                                true, 1, 100);

    mParameterSet->addParameter(QString("randomSearchIter"),
                                1,
                                QString("randomSearchIter"),
                                QString("Number of random search iterations"),
                                true, 1, 100);

    mParameterSet->setName(QString("Laplacian Image Inpainting algorithm"));
    mParameterSet->setLabel(QString("Laplacian Image Inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Parameters settings"));
}


void LaplacianImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getOriginalImage();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }
    if (image.channels() == 4)
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    image = universalConvertTo(image, CV_8UC3);

    cv::Mat mask = ~mDataManager->getMask();
    if (mask.empty())
    {
        qWarning() << "The mask is empty";
        return;
    }
    if (mask.channels() == 4)
        cv::cvtColor(mask, mask, cv::COLOR_BGRA2BGR);
    mask = universalConvertTo(mask, CV_8UC1);

    int patchSize, minSize, numEM, decreaseFactor, minEMIter, randomSearchIter, maxIters;
    double distanceWeight, distanceMetric, a, b;
    if (!mParameterSet->getValue(QString("patchSize"), patchSize) ||
        !mParameterSet->getValue(QString("distanceWeight"), distanceWeight) ||
        !mParameterSet->getValue(QString("distanceMetric"), distanceMetric) ||
        !mParameterSet->getValue(QString("minSize"), minSize) ||
        !mParameterSet->getValue(QString("numEM"), numEM) ||
        !mParameterSet->getValue(QString("decreaseFactor"), decreaseFactor) ||
        !mParameterSet->getValue(QString("minEMIter"), minEMIter) ||
        !mParameterSet->getValue(QString("randomSearchIter"), randomSearchIter))
    {
        qWarning() << "Could not retrieve all parameters.";
        return;
    }

    emit sendOtherMessage("Processing image...");
    cv::Mat inpainted = laplacianInpaint(image, mask, patchSize, distanceWeight, distanceMetric,
                                         minSize, numEM, decreaseFactor, minEMIter, randomSearchIter);
    emit sendOtherMessage("");

    mDataManager->setImage(image);
    mDataManager->setInpaintedImage(inpainted);

    emit sendImageProcess(inpainted);
}

cv::Mat LaplacianImageInpainting::laplacianInpaint(cv::Mat colormat, cv::Mat maskmat, int patchSize, double distanceWeight, double distanceMetric, int minSize, int numEM, int decreaseFactor, int minEMIter, int randomSearchIter)
{
    cv::Mat origcolormat, rgbmat;
    double *colorptr, *maskptr;
    int height, width;

    int decrease_factor;
    int min_iter;
    char *outputfilename, *fname, *processfilename, *dirname;

    //inpainting parameter
    int num_em;
    int psz;
    int min_size;
    int rs_iter;
    double gamma;
    double lambda;

    //pyramid
    //gpyr - Gaussian pyramid
    //upyr - upsampled Gaussian pyramid
    //fpyr - Laplacian pyramid
    std::vector<std::pair<int,int> > pyr_size;
    std::vector<cv::Mat> mask_gpyr, color_gpyr;
    std::vector<cv::Mat> mask_upyr, color_upyr;
    std::vector<cv::Mat> mask_fpyr, color_fpyr;
    std::vector<cv::Mat> rgb_gpyr,rgb_fpyr,rgb_upyr;

    //Laplacian inpainting object
    LaplacianInpainting inpainting;

    processfilename = (char*)malloc(sizeof(char) * 200);
    dirname = (char*)malloc(sizeof(char) *200);
    fname = (char*)malloc(sizeof(char) *200);
    outputfilename = (char*)malloc(sizeof(char) *200);

    ////////////////////////
    //*Step 1: read input*//
    ////////////////////////

    psz             = patchSize;        // patch size
    gamma           = distanceWeight;   // gamma - distance weight parameter. [Wexler et al. 2007]
    min_size        = minSize;          // minimum size
    lambda          = distanceMetric;   // lambda - ratio btw lab Laplacian patch distance and lab upsampled Gaussian patch distance - distance metric parameter
    num_em          = numEM;            // the number of EM iteration
    decrease_factor = decreaseFactor;   // decrease_factor
    min_iter        = minEMIter;        // minimum iteration
    rs_iter         = randomSearchIter; // random search iteration

    width = colormat.cols;  //image width
    height = colormat.rows; //image height

    int tmp_width = width,tmp_height = height;
    int tmp = 1;
    for(int i=0;;i++)
    {
        tmp_width  >>= 1;
        tmp_height >>= 1;
        if(min_size > tmp_width || min_size > tmp_height)
            break;
        tmp <<= 1;
    }

    if(width%tmp) width=width-(width%tmp);
    if(height%tmp) height=height-(height%tmp);

    origcolormat = colormat.clone();

    colormat = colormat(cv::Rect(0,0,width, height));  //crop the image
    maskmat = maskmat(cv::Rect(0,0,width, height));


    colormat.convertTo(colormat, CV_32FC3);   //convert an uchar image to a float image (Input of cvtColor function should be a single precision )
    maskmat.convertTo(maskmat,CV_64FC1);      //double mask

    colormat/=255.0;	//255 -> 1.0
    maskmat/=255.0;

    colormat.convertTo(rgbmat, CV_64FC3);

    //convert rgb to CIEL*a*b*
    cv::cvtColor(colormat, colormat, cv::COLOR_RGB2Lab); //RGB to Lab
    colormat.convertTo(colormat, CV_64FC3);   //single -> double

    //values in mask region should be zero.
    colorptr = (double*) colormat.data;
    maskptr = (double*) maskmat.data;

    //refine mask and color image
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            int ndx = i*width + j;
            if(maskptr[ndx]>0){
                colorptr[3*ndx] = 0;
                colorptr[3*ndx+1] = 0;
                colorptr[3*ndx+2] = 0;
                maskptr[ndx]=1;
            }
            else maskptr[ndx]=0;
        }
    }

    ///////////////////////////////////
    //*step 2: set parameters       *//
    ///////////////////////////////////
    inpainting.gamma_ = gamma;            //parameter for voting
    inpainting.lambda_ = lambda;          //ratio between Laplacian patch distance metric and upsampled Gaussian patch distance metric.
    inpainting.minsize_ = min_size;       //minimum scale
    inpainting.psz_ = psz;                //patch size
    inpainting.highconfidence_ = 1.0f;   //confidence for non-mask region
    inpainting.patchmatch_iter_ = 10;   //EM iteration
    inpainting.siminterval_ = 3.0f;     //parameter for voting
    inpainting.rs_iter_ = rs_iter;      //random search itertation

    ///////////////////////////////////
    //*step 3: generate pyramid     *//
    ///////////////////////////////////

    inpainting.constructLaplacianPyr(rgb_gpyr, rgb_upyr, rgb_fpyr, rgbmat);

    //construct Laplacian pyramid
    inpainting.constructLaplacianPyr(color_gpyr, color_upyr, color_fpyr, colormat);
    inpainting.constructLaplacianPyr(mask_gpyr, mask_upyr, mask_fpyr, maskmat);

    //reverse order (from low-res to high-res)
    std::reverse(color_gpyr.begin(), color_gpyr.end());
    std::reverse(color_upyr.begin(), color_upyr.end());
    std::reverse(color_fpyr.begin(), color_fpyr.end());
    std::reverse(mask_gpyr.begin(), mask_gpyr.end());
    std::reverse(mask_upyr.begin(), mask_upyr.end());
    std::reverse(mask_fpyr.begin(), mask_fpyr.end());

    //compute pyr_size
    pyr_size.clear();

    //set size
    for(size_t i = 0; i < color_gpyr.size(); i++)
        pyr_size.push_back(std::pair<int,int>(color_gpyr[i].rows, color_gpyr[i].cols));

    //refine mask
    fixDownsampledMaskMatColorMat(mask_gpyr[0],color_gpyr[0]);

    for (size_t i = 0; i < mask_upyr.size(); i++)
    {
        fixDownsampledMaskMatColorMat(mask_upyr[i],color_upyr[i]);
        fixDownsampledMaskMatColorMat(mask_gpyr[i+1],color_gpyr[i+1]);
        color_fpyr[i]=color_gpyr[i+1]-color_upyr[i];

        mask_upyr[i]=mask_gpyr[i+1]+mask_upyr[i];
        fixDownsampledMaskMat(mask_upyr[i]);
        fixDownsampledMaskMatColorMat(mask_upyr[i],color_upyr[i]);
        fixDownsampledMaskMatColorMat(mask_upyr[i],color_gpyr[i+1]);

//        displayMat<double>(mask_upyr[i]-mask_gpyr[i+1],"gpyr",cv::Rect(0,0,mask_gpyr[i+1].cols,mask_gpyr[i+1].rows));
//        displayMat<double>(mask_upyr[i],"upyr",cv::Rect(0,0,mask_upyr[i].cols,mask_upyr[i].rows));
//        cv::waitKey();
//        if(i<mask_upyr.size()-1)
//            cv::pyrUp(mask_upyr[i],mask_upyr[i+1],cv::Size(mask_upyr[i+1].cols,mask_upyr[i+1].rows));
    }

    //dilate mask?

    /////////////////////////////////////////////
    //*step 4: initialize the zero level image*//
    /////////////////////////////////////////////

    cv::Mat color8u, mask8u, feature8u;
    cv::Mat repmask;
    cv::Mat trg_color;
    cv::Mat trg_feature;

    double featuremin, featuremax;
    cv::minMaxLoc(color_fpyr[0], &featuremin, &featuremax);

    color_upyr[0].convertTo(color8u,CV_32FC3);
    cv::cvtColor(color8u, color8u, cv::COLOR_Lab2RGB);

    color8u = color8u*255.;
    mask8u = mask_upyr[0]*255.;

    feature8u = (color_fpyr[0]-featuremin)/(featuremax-featuremin) * 255.;

    color8u.convertTo(color8u, CV_8U);
    mask8u.convertTo(mask8u, CV_8U);
    feature8u.convertTo(feature8u, CV_8U);

    //initialization
    //We use a Navier-Stokes based method [Navier et al. 01] only for initialization.
    //http://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf
    cv::inpaint(color8u, mask8u, color8u, 10, cv::INPAINT_NS);
    cv::inpaint(feature8u, mask8u, feature8u, 10, cv::INPAINT_NS);

    color8u.convertTo(color8u,CV_32FC3);
    color8u=color8u/255.f;
    cv::cvtColor(color8u, color8u, cv::COLOR_RGB2Lab);
    color8u.convertTo(color_upyr[0],CV_64FC3);
    feature8u.convertTo(color_fpyr[0],CV_64FC3);
    color_fpyr[0] = color_fpyr[0] / 255.0 * (featuremax-featuremin) + featuremin;
    //	depth8u.convertTo(depth_gpyr[0],CV_64FC1);

    trg_color = color_upyr[0].clone();
    trg_feature = color_fpyr[0].clone();

    //displayMat<double>(trg_feature,"feature",cv::Rect(0,0,trg_feature.cols, trg_feature.rows));
    //displayLABMat(trg_color,"color",cv::Rect(0,0,trg_color.cols, trg_color.rows));
    int cur_iter = num_em;

    /////////////////////////////////
    //*Step 5: Do image completion*//
    /////////////////////////////////

    cv::Mat nnf, nnff;
    cv::Mat nnferr;
    cv::Mat nxt_color;
    bool *patch_type = NULL;

    nnf = cv::Mat::zeros(pyr_size[1].first, pyr_size[1].second, CV_32SC2); // H x W x 2 int

    clock_t t;
    clock_t recont,accumt;
    int f;
    accumt=0;
    t=clock();

    for(size_t ilevel = 0; ilevel < color_upyr.size(); ilevel++)
    {
        emit sendOtherMessage("Processing level " + QString::number(ilevel) + " of " + QString::number(color_upyr.size()) + "...");

        if(ilevel){

            //resize trg_color, trg_depth, trg_feature
            recont = clock();
            nxt_color = trg_color + trg_feature; //Gaussian = upsampled Gaussian + Laplacian
            recont = clock()-recont;
            accumt+=recont;

            cv::pyrUp(nxt_color, trg_color, cv::Size(trg_color.cols * 2, trg_color.rows * 2)); // upsample a low-level Gaussian image
            cv::pyrUp(trg_feature, trg_feature, cv::Size(trg_feature.cols * 2, trg_feature.rows * 2)); //upsample a Laplacian image (we will reset a initial laplacian image later)

            double *trgcptr = (double*) trg_color.data;
            double *trgfptr = (double*) trg_feature.data;
            double *maskptr = (double*) mask_upyr[ilevel].data;

            //initialize
            for(int i = 0; i < pyr_size[ilevel+1].first; i++)
            {
                for(int j = 0; j < pyr_size[ilevel+1].second; j++)
                {
                    int ndx = i * pyr_size[ilevel+1].second + j;
                    if(maskptr[ndx] < 0.1)
                    {
                        trgcptr[3 * ndx] = ((double*)(color_upyr[ilevel].data))[3 * ndx];
                        trgcptr[3 * ndx + 1] = ((double*)(color_upyr[ilevel].data))[3 * ndx + 1];
                        trgcptr[3 * ndx + 2] = ((double*)(color_upyr[ilevel].data))[3 * ndx + 2];
                        trgfptr[3 * ndx] = ((double*)(color_fpyr[ilevel].data))[3 * ndx];
                        trgfptr[3 * ndx + 1] = ((double*)(color_fpyr[ilevel].data))[ 3 * ndx + 1];
                        trgfptr[3 * ndx + 2] = ((double*)(color_fpyr[ilevel].data))[3 * ndx + 2];
                    }
                }
            }

            //NNF propagation
            recont = clock();
            inpainting.upscaleImages(nnf, nnferr, patch_type, trg_feature, mask_upyr[ilevel-1].clone(), mask_upyr[ilevel].clone());
            recont = clock() - recont;
            accumt += recont;


            //upscale NNF field
            nnf.convertTo(nnff, CV_64FC2);
            cv::resize(nnff, nnff, cv::Size(pyr_size[ilevel + 1].second, pyr_size[ilevel + 1].first),cv::INTER_LINEAR);
            nnff.convertTo(nnf, CV_32SC2);
            nnff = nnf * 2;
        }

        if(patch_type != NULL)
            free(patch_type);
        patch_type = (bool*)malloc(sizeof(bool) * pyr_size[ilevel + 1].first * pyr_size[ilevel + 1].second);

        nnferr = cv::Mat::zeros(pyr_size[ilevel + 1].first, pyr_size[ilevel + 1].second, CV_64FC1); // H x W x 1 double

        //do EM iteration
        inpainting.doEMIterLap(nnf, nnferr, patch_type, trg_color, trg_feature, mask_upyr[ilevel].clone(), pyr_size[ilevel+1], cur_iter, cv::Size(width, height), processfilename);

        //compute next iteration
        cur_iter -= decrease_factor;
        if(cur_iter < min_iter)
            cur_iter = min_iter;

        {
            cv::Mat tmpimg;
            tmpimg = trg_color.clone() + trg_feature.clone();
            tmpimg.convertTo(tmpimg, CV_32FC3);
            cv::cvtColor(tmpimg, tmpimg, cv::COLOR_Lab2RGB);
            tmpimg=255*tmpimg;
            tmpimg.convertTo(tmpimg, CV_8UC3);
            emit sendImageProcess(tmpimg);
        }
    }

    //print final result
    cv::Mat tmpimg;

    tmpimg = trg_color.clone() + trg_feature.clone();
    tmpimg.convertTo(tmpimg, CV_32FC3);
    cv::cvtColor(tmpimg, tmpimg, cv::COLOR_Lab2RGB);
    tmpimg=255*tmpimg;
    tmpimg.convertTo(tmpimg, CV_8UC3);
    //		displayMat<double>(tmpimg,outputfilename,cv::Rect(0,0,tmpimg.cols, tmpimg.rows));

    free(processfilename);
    free(dirname);
    free(fname);
    free(outputfilename);

    return tmpimg;
}

void LaplacianImageInpainting::fixDownsampledMaskMat(cv::Mat mask)
{
    double TT = 0.6;
    double *maskptr = (double*) mask.data;

    for(int i=0;i<mask.rows;i++){
        for(int j=0;j<mask.cols;j++){
            int ndx = i*mask.cols+j;
            if(maskptr[ndx]>TT){
                maskptr[ndx]=1;
            }
            else{
                maskptr[ndx]=0;
            }
        }
    }
}

void LaplacianImageInpainting::fixDownsampledMaskMatColorMat(cv::Mat mask, cv::Mat color)
{
    double TT = 0.6;
    double *maskptr = (double*) mask.data;
    double *colorptr = (double*) color.data;

    for(int i=0;i<mask.rows;i++){
        for(int j=0;j<mask.cols;j++){
            int ndx = i*mask.cols+j;
            if(maskptr[ndx]>TT){
                maskptr[ndx]=1;
                colorptr[3*ndx]=0;
                colorptr[3*ndx+1]=0;
                colorptr[3*ndx+2]=0;
            }
            else{
                colorptr[3*ndx]=colorptr[3*ndx]/(1-maskptr[ndx]);
                colorptr[3*ndx+1]=colorptr[3*ndx+1]/(1-maskptr[ndx]);
                colorptr[3*ndx+2]=colorptr[3*ndx+2]/(1-maskptr[ndx]);
                maskptr[ndx]=0;
            }
        }
    }
}
void LaplacianInpainting::constructLaplacianPyrMask(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat mask,cv::Mat &img){

    cv::Mat prvimg, curimg, curfimg, upimg;
    cv::Mat prvmask, curmask, upmask;
    gpyr.push_back(img);
    prvimg = img;
    prvmask = mask;

    for(;prvimg.cols>=2*minsize_&&prvimg.rows>=2*minsize_;){
        cv::pyrDown(prvimg, curimg, cv::Size(prvimg.cols/2, prvimg.rows/2));
        cv::pyrUp(curimg, upimg, cv::Size(curimg.cols*2, curimg.rows*2));
        cv::pyrDown(prvmask, curmask, cv::Size(prvimg.cols/2, prvimg.rows/2));
        cv::pyrUp(curmask, upmask, cv::Size(curimg.cols*2, curimg.rows*2));

        //		fixDownsampledMaskColorMat(upmask, upimg);
        //		fixDownsampledMaskColorMat(curmask, curimg);

        curfimg = prvimg - upimg;
        gpyr.push_back(curimg);
        upyr.push_back(upimg);
        fpyr.push_back(curfimg);
        //displayLABMat(curimg, "gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
        //displayLABMat(upimg, "up gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
        //displayLABMat(curfimg, "lap", cv::Rect(0, 0, curimg.rows, curimg.cols));
        prvimg=curimg;
        prvmask=curmask;
    }
}

void LaplacianInpainting::constructLaplacianPyr(std::vector<cv::Mat> &gpyr, std::vector<cv::Mat> &upyr, std::vector<cv::Mat> &fpyr,cv::Mat &img){

    cv::Mat prvimg, curimg, curfimg, upimg;
    gpyr.push_back(img.clone());
    prvimg = img.clone();

    for(;prvimg.cols>=2*minsize_&&prvimg.rows>=2*minsize_;){
        cv::pyrDown(prvimg, curimg, cv::Size(prvimg.cols/2, prvimg.rows/2));
        cv::pyrUp(curimg, upimg, cv::Size(curimg.cols*2, curimg.rows*2));
        curfimg = prvimg - upimg;
        gpyr.push_back(curimg);
        upyr.push_back(upimg);
        fpyr.push_back(curfimg);
        //displayLABMat(curimg, "gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
        //displayLABMat(upimg, "up gaussian", cv::Rect(0, 0, curimg.rows, curimg.cols));
        //displayLABMat(curfimg, "lap", cv::Rect(0, 0, curimg.rows, curimg.cols));
        prvimg=curimg;
    }
}

void LaplacianInpainting::findNearestNeighborLap(cv::Mat nnf,cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size,int emiter){

    /*Patch preparation*/
    std::vector<double*> colorpatches,colorfpatches;

    srand(time(NULL));

    double *maskptr = (double*) maskmat.data;
    double *colorptr = (double*) colormat.data;
    double *colorfptr = (double*) colorfmat.data;

    double errmin, errmax;

    int tmph = size.first - psz_ + 1; //we ignore
    int tmpw = size.second - psz_ + 1;
    int randomcnt=0, propagationcnt=0;
    int lurow,lucol,rdrow, rdcol;

    lurow = tmph;
    lucol = tmpw;
    rdrow = 0;
    rdcol = 0;


    double *colorpatch, *colorfpatch;

    //collect patches
    for(int i=0;i<tmph;i++){
        for(int j=0;j<tmpw;j++){
            int ndx = i * size.second + j;

            int flag=0;
            colorpatch  = (double*)malloc(sizeof(double) * psz_ * psz_ * 3);
            colorfpatch  = (double*)malloc(sizeof(double) * psz_ * psz_ * 3);

            //copy patch
            for(int i2=0;i2<psz_;i2++){
                for(int j2=0;j2<psz_;j2++){
                    int ndx2 = (i+i2) * size.second + (j+j2);
                    int pndx = i2 * psz_ + j2;
                    if(maskptr[ndx2]>0.00)
                        flag=1;

                    colorpatch [3 * pndx	] = colorptr[3 * ndx2];
                    colorpatch [3 * pndx + 1] = colorptr[3 * ndx2 + 1];
                    colorpatch [3 * pndx + 2] = colorptr[3 * ndx2 + 2];
                    colorfpatch [3 * pndx	] = colorfptr[3 * ndx2];
                    colorfpatch [3 * pndx + 1] = colorfptr[3 * ndx2 + 1];
                    colorfpatch [3 * pndx + 2] = colorfptr[3 * ndx2 + 2];
                }
            }

            if(flag){ // find bounding box
                rdrow = std::max(rdrow, i);
                rdcol = std::max(rdcol, j);
                lurow = std::min(lurow, i);
                lucol = std::min(lucol, j);
            }

            patch_type [ndx] = flag;//If variable flag is one, there is a mask pixel in the patch.
            colorpatches.push_back(colorpatch);//Note that index for patches is i*tmpw + j since we only take inner patches.
            colorfpatches.push_back(colorfpatch);//feature patch
        }
    }

    rdrow = std::min(rdrow+2*psz_, tmph-1);
    rdcol = std::min(rdcol+2*psz_, tmpw-1);
    lurow = std::max(lurow-2*psz_, 0);
    lucol = std::max(lucol-2*psz_, 0);

    nnfcount_=(rdcol-lucol)*(rdrow-lurow);

    int* nnfptr = (int*)nnf.data;
    double* nnferrptr = (double*)nnferr.data;

    /*Initialize NNF*/
    for(int i=lurow;i<=rdrow;i++){
        for(int j=lucol;j<=rdcol;j++){
            int ndx = i * size.second + j;
            int newrow, newcol;
            double newerr;

            //nnferrptr[ndx] = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, nnfptr[ndx*2] * tmpw +nnfptr[ndx*2 + 1], psz_, lambda_);
            nnferrptr[ndx] = 50000;

            do{
                newrow = rand() % tmph;//row
                newcol = rand() % tmpw;//col
            }while(patch_type[newrow*size.second+newcol]||(newrow==i&&newcol==j));//Until patch is from a source patch. If the pointed patch is a target, reset values.

            newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, newrow * tmpw + newcol, psz_, lambda_);

            if(emiter == -1){
                nnfptr[ndx*2	]=newrow;//row
                nnfptr[ndx*2 + 1]=newcol;//col
                nnferrptr[ndx] = newerr;
            }
            else{
                if(nnferrptr[ndx] >newerr || patch_type[nnfptr[ndx*2]*size.second+nnfptr[ndx*2+1]]){
                    nnfptr[ndx*2	]=newrow;//row
                    nnfptr[ndx*2 + 1]=newcol;//col
                    nnferrptr[ndx] = newerr;
                }
            }
        }
    }

    //cv::minMaxLoc(nnferr, &errmin, &errmax);
    //std::cout << "max error: " << errmax << std::endl;
    //cv::imshow("nnf error", nnferr/errmax);
    //cv::waitKey();
    /*Patchmatch start*/

    for(int patchiter = 0; patchiter < patchmatch_iter_; patchiter++){

        /*random search*/
        for(int i=lurow;i<=rdrow;i++){
            for(int j=lucol;j<=rdcol;j++){

                int vrow, vcol;
                int ndx = i*size.second + j;
                int w_row = tmph, w_col = tmpw;
                double alpha = 0.5;
                int cur_row, cur_col;
                double newerr;
                int row1, row2, col1, col2;
                int ranr, ranc;

                vrow = nnfptr[ndx*2];
                vcol = nnfptr[ndx*2+1];

                cur_row = w_row;
                cur_col = w_col;

                for(int h=0;cur_row>=1&&cur_col>=1;h++){
                    //
                    row1 = vrow - cur_row;
                    row2 = vrow + cur_row+1;
                    col1 = vcol - cur_col;
                    col2 = vcol + cur_col+1;

                    //cropping
                    if(row1<0) row1 = 0;
                    if(row2>tmph) row2 = tmph;
                    if(col1<0) col1 = 0;
                    if(col2>tmpw) col2 = tmpw;

                    for(int k = 0 ; k < rs_iter_;k++){

                        do{
                            ranr = (rand() % (row2 - row1)) + row1;//2~4 2,5	 3	0,1,2 + 2
                            ranc = (rand() % (col2 - col1)) + col1;
                        }while(patch_type[ranr * size.second + ranc]);

                        newerr =  computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, ranr * tmpw + ranc, psz_, lambda_);

                        if(newerr < nnferrptr[ndx]){
                            randomcnt++;
                            nnfptr[ndx*2	] = ranr;//row
                            nnfptr[ndx*2 + 1] = ranc;//col
                            nnferrptr[ndx]	= newerr;
                        }
                    }
                    //shrink a window size
                    cur_row >>= 1;
                    cur_col >>= 1;
                }
            }
        }

        if(patchiter&1){//odd leftup order

            for(int i=rdrow;i>=lurow;i--){
                for(int j=rdcol;j>=lucol;j--){
                    //			for(int i=tmph-1;i>=0;i--){
                    //				for(int j=tmpw-1;j>=0;j--){

                    int vrow, vcol;
                    int ndx = i*size.second + j;
                    int w_row = tmph, w_col = tmpw;
                    double alpha = 0.5;
                    int cur_row, cur_col;
                    double newerr;
                    int row1, row2, col1, col2;
                    int ranr, ranc;

                    int vrowright, vcolright;
                    int vrowdown, vcoldown;

                    /*propagation*/
                    if(j<rdcol){//left
                        vrowright = nnfptr[ndx * 2 + 2];
                        vcolright = nnfptr[ndx * 2 + 3];
                        if(vcolright>0)
                            --vcolright;

                        if(!patch_type[vrowright*size.second + vcolright]){

                            newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowright * tmpw + vcolright, psz_, lambda_);

                            if(newerr < nnferrptr[ndx]){
                                propagationcnt++;
                                nnfptr[ndx*2	] = vrowright;//row
                                nnfptr[ndx*2 + 1] = vcolright;//col
                                nnferrptr[ndx]	= newerr;
                            }
                        }
                    }

                    if(i<rdrow){//right
                        vrowdown = nnfptr[(ndx+size.second) * 2	];
                        vcoldown = nnfptr[(ndx+size.second) * 2 + 1];
                        if(vrowdown>0)
                            --vrowdown;
                        if(!patch_type[vrowdown*size.second + vcoldown]){

                            newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowdown * tmpw + vcoldown, psz_, lambda_);

                            if(newerr < nnferrptr[ndx]){
                                propagationcnt++;
                                nnfptr[ndx*2	] = vrowdown;//row
                                nnfptr[ndx*2 + 1] = vcoldown;//col
                                nnferrptr[ndx]	= newerr;
                            }
                        }
                    }
                }
            }
        }
        else{//even

            for(int i=lurow;i<=rdrow;i++){
                for(int j=lucol;j<=rdcol;j++){
                    //			for(int i=0;i<tmph;i++){//right down order
                    //				for(int j=0;j<tmpw;j++){

                    int vrow, vcol;
                    int ndx = i*size.second + j;
                    int w_row = tmph, w_col = tmpw;
                    double alpha = 0.5;
                    int cur_row, cur_col;
                    double newerr;
                    int row1, row2, col1, col2;
                    int ranr, ranc;

                    int vrowleft, vcolleft;
                    int vrowup, vcolup;

                    /*propagation*/
                    if(j>lucol){//left
                        vrowleft = nnfptr[ndx * 2 - 2];
                        vcolleft = nnfptr[ndx * 2 - 1];
                        if(vcolleft<tmpw-1)
                            ++vcolleft;

                        if(!patch_type[vrowleft*size.second + vcolleft]){

                            newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowleft * tmpw + vcolleft, psz_, lambda_);

                            if(newerr < nnferrptr[ndx]){
                                propagationcnt++;
                                nnfptr[ndx*2	] = vrowleft;//row
                                nnfptr[ndx*2 + 1] = vcolleft;//col
                                nnferrptr[ndx]	= newerr;
                            }
                        }
                    }

                    if(i>lurow){//up
                        vrowup = nnfptr[(ndx-size.second) * 2	];
                        vcolup = nnfptr[(ndx-size.second) * 2 + 1];
                        if(vrowup<tmph-1)
                            ++vrowup;
                        if(!patch_type[vrowup*size.second + vcolup]){

                            newerr = computePatchErrorLap(colorpatches, colorfpatches, i*tmpw+j, vrowup * tmpw + vcolup, psz_, lambda_);

                            if(newerr < nnferrptr[ndx]){
                                propagationcnt++;
                                nnfptr[ndx*2	] = vrowup;//row
                                nnfptr[ndx*2 + 1] = vcolup;//col
                                nnferrptr[ndx]	= newerr;
                            }
                        }
                    }
                }
            }
        }
    }

    while(!colorpatches.empty()){
        free(colorpatches.back());
        colorpatches.pop_back();
    }
    while(!colorfpatches.empty()){
        free(colorfpatches.back());
        colorfpatches.pop_back();
    }
}

void LaplacianInpainting::colorVoteLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size){

    int tmph = size.first - psz_ + 1;
    int tmpw = size.second - psz_ + 1;

    cv::Mat weight;
    cv::Mat colorsum, colorfsum;
    cv::Mat dist;
    cv::Mat similarity;
    cv::Mat squarednnferr;
    double nnfavg, nnfsqavg, variance;
    double *nnferrptr = (double*) nnferr.data;
    double maskcnt = 0;

    nnfavg = 0;
    nnfsqavg = 0;

    nnfavg = cv::sum(nnferr).val[0];
    cv::multiply(nnferr,nnferr,squarednnferr);
    nnfsqavg = cv::sum(squarednnferr).val[0];

    nnfavg /= nnfcount_;
    nnfsqavg /= nnfcount_;
    variance = nnfsqavg - nnfavg * nnfavg;

    //	std::cout << "variance: " << variance << std::endl;

    //Wexler's similarity function
    //cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance)) * (nnfavg + 0.68 * sqrt(variance)) * siminterval_), similarity);//0.68 percentile

    //ours
    cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance))  * siminterval_), similarity);//0.68 percentile

    double *colorptr = (double*) colormat.data;
    double *colorfptr = (double*) colorfmat.data;
    double *similarityptr = (double*) similarity.data;

    maskmat.convertTo(maskmat, CV_8UC1);
    cv::distanceTransform(maskmat, dist, cv::DIST_L1, 3);

    //Wexler's distance-weight function
    //dist = dist * log(gamma_) * -1;
    //cv::exp(dist, dist);

    //ours
    pow(dist, -1 * gamma_, dist);

    weight = cv::Mat::zeros(size.first, size.second, CV_64FC1);
    colorsum = cv::Mat::zeros(size.first, size.second, CV_64FC3);
    colorfsum = cv::Mat::zeros(size.first, size.second, CV_64FC3);

    double *weightptr = (double*) weight.data;
    double *colorsumptr = (double*) colorsum.data;
    double *colorfsumptr = (double*) colorfsum.data;
    float *distptr = (float*) dist.data;
    int *nnfptr = (int*) nnf.data;
    unsigned char *maskptr = (unsigned char*) maskmat.data;

    for(int i=0;i<size.first;i++){
        for(int j=0;j<size.second;j++){
            int ndx = i*size.second + j;
            if(maskptr[ndx]==0)
                distptr[ndx]=highconfidence_;
        }
    }

    for(int i=0;i<tmph;i++){
        for(int j=0;j<tmpw;j++){
            int ndx= i*size.second+j;
            int patchcenter_ndx= (i+(psz_>>1))*size.second+j+(psz_>>1);
            double alpha = 0.0;
#ifdef CENTERINMASK
            if(maskptr[patchcenter_ndx]>0.0){
#else
            if(patch_type[ndx]){//If a patch is a target patch
#endif
                alpha = distptr[patchcenter_ndx];
                //pixel by pixel
                for(int i2=0;i2<psz_;i2++){
                    for(int j2=0;j2<psz_;j2++){
                        int ndx2 = (i+i2)*size.second + (j+j2);
                        int ndx3 = (nnfptr[2*ndx]+i2) * size.second + nnfptr[2*ndx+1]+j2;

                        weightptr[ndx2] += alpha * similarityptr[ndx];
                        colorsumptr[3*ndx2] += alpha * similarityptr[ndx] * colorptr[3*ndx3	];
                        colorsumptr[3*ndx2+1] += alpha * similarityptr[ndx] * colorptr[3*ndx3+1];
                        colorsumptr[3*ndx2+2] += alpha * similarityptr[ndx] * colorptr[3*ndx3+2];
                        colorfsumptr[3*ndx2] += alpha * similarityptr[ndx] * colorfptr[3*ndx3	];
                        colorfsumptr[3*ndx2+1] += alpha * similarityptr[ndx] * colorfptr[3*ndx3+1];
                        colorfsumptr[3*ndx2+2] += alpha * similarityptr[ndx] * colorfptr[3*ndx3+2];
                    }
                }
            }
        }
    }


    //normalize
    for(int i=0;i<size.first;i++){
        for(int j=0;j<size.second;j++){
            int ndx = i*size.second +j;
            if(maskptr[ndx]>0.0){
                colorptr [3*ndx]   = colorsumptr[3*ndx] / weightptr[ndx];
                colorptr [3*ndx+1] = colorsumptr[3*ndx+1] / weightptr[ndx];
                colorptr [3*ndx+2] = colorsumptr[3*ndx+2] / weightptr[ndx];
                colorfptr [3*ndx]   = colorfsumptr[3*ndx] / weightptr[ndx];
                colorfptr [3*ndx+1] = colorfsumptr[3*ndx+1] / weightptr[ndx];
                colorfptr [3*ndx+2] = colorfsumptr[3*ndx+2] / weightptr[ndx];
            }
        }
    }
    //	cv::imshow("color", colormat);
    //	cv::waitKey();

}

void LaplacianInpainting::upscaleImages(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colorfmat,  cv::Mat dmaskmat,  cv::Mat umaskmat){

    std::pair<int, int> dsize (nnf.rows, nnf.cols), usize(colorfmat.rows, colorfmat.cols);

    int dtmph = dsize.first - psz_ + 1;
    int dtmpw = dsize.second - psz_ + 1;

    cv::Mat weight;
    cv::Mat colorfsum;
    cv::Mat dist;
    cv::Mat similarity;
    cv::Mat squarednnferr;
    double nnfavg, nnfsqavg, variance;
    double *nnferrptr = (double*) nnferr.data;
    double maskcnt = 0;

    nnfavg = 0;
    nnfsqavg = 0;

    nnfavg = cv::sum(nnferr).val[0];
    cv::multiply(nnferr,nnferr,squarednnferr);
    nnfsqavg = cv::sum(squarednnferr).val[0];

    nnfavg /= nnfcount_;
    nnfsqavg /= nnfcount_;
    variance = nnfsqavg - nnfavg * nnfavg;


    //Wexler's similarity function
    //cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance)) * (nnfavg + 0.68 * sqrt(variance)) * siminterval_), similarity);//0.68 percentile

    //ours
    cv::exp( - nnferr / (2.0 * (nnfavg + 0.68 * sqrt(variance))  * siminterval_), similarity);//0.68 percentile

    double *colorfptr = (double*) colorfmat.data;
    double *similarityptr = (double*) similarity.data;

    dmaskmat.convertTo(dmaskmat, CV_8UC1);
    cv::distanceTransform(dmaskmat, dist, cv::DIST_L1, 3);
    //Wexler's distance-weight function
    //dist = dist * log(gamma_) * -1;
    //cv::exp(dist, dist);

    //ours
    pow(dist, -1 * gamma_, dist);

    weight = cv::Mat::zeros(usize.first, usize.second, CV_64FC1);
    colorfsum = cv::Mat::zeros(usize.first, usize.second, CV_64FC3);

    double *weightptr = (double*) weight.data;
    double *colorfsumptr = (double*) colorfsum.data;
    float *distptr = (float*) dist.data;
    int *nnfptr = (int*) nnf.data;
    unsigned char *dmaskptr = (unsigned char*) dmaskmat.data;
    double *umaskptr = (double*) umaskmat.data;

    for(int i=0;i<dsize.first;i++){
        for(int j=0;j<dsize.second;j++){
            int ndx = i*dsize.second + j;
            if(!dmaskptr[ndx])
                distptr[ndx]=highconfidence_;
        }
    }

    for(int i=0;i<dtmph;i++){
        for(int j=0;j<dtmpw;j++){
            int dndx= i*dsize.second+j;
            int undx= (2*i) * usize.second + 2*j;
            int patchcenter_dndx= (i+(psz_>>1))*dsize.second+j+(psz_>>1);
            double alpha = 0.0;
#ifdef CENTERINMASK
            if(dmaskptr[patchcenter_dndx]>0.0){
#else
            if(patch_type[dndx]){//If a patch is a target patch
#endif
                alpha = distptr[patchcenter_dndx];
                //pixel by pixel
                for(int i2=0;i2<psz_*2;i2++){
                    for(int j2=0;j2<psz_*2;j2++){
                        int undx2 = (2*i+i2)*usize.second + (2*j+j2);
                        int undx3 = (2*nnfptr[2*dndx]+i2) * usize.second + 2 * nnfptr[2*dndx+1]+j2;

                        weightptr[undx2] += alpha * similarityptr[dndx];
                        colorfsumptr[3*undx2] += alpha * similarityptr[dndx] * colorfptr[3*undx3	];
                        colorfsumptr[3*undx2+1] += alpha * similarityptr[dndx] * colorfptr[3*undx3+1];
                        colorfsumptr[3*undx2+2] += alpha * similarityptr[dndx] * colorfptr[3*undx3+2];

                    }
                }
            }
        }
    }

    //normalize
    for(int i=0;i<usize.first;i++){
        for(int j=0;j<usize.second;j++){
            int undx = i*usize.second +j;
            if(umaskptr[undx]>0.0){
                colorfptr [3*undx]   = colorfsumptr[3*undx] / weightptr[undx];
                colorfptr [3*undx+1] = colorfsumptr[3*undx+1] / weightptr[undx];
                colorfptr [3*undx+2] = colorfsumptr[3*undx+2] / weightptr[undx];
            }
        }
    }
}

void LaplacianInpainting::doEMIterLap(cv::Mat nnf, cv::Mat nnferr, bool *patch_type, cv::Mat colormat, cv::Mat colorfmat, cv::Mat maskmat, std::pair<int, int> size, int num_emiter, cv::Size orig_size, char *processfilename){
    double errmin, errmax;
    char *outputfilename;

    outputfilename = (char*) malloc(sizeof(char) * 300);

    cv::Mat a, tmpimg,tmpimg2,tmpimg1;

    for(int emiter = 0; emiter< num_emiter; emiter++){
        //compute the nearest neighbor fields
        findNearestNeighborLap(nnf, nnferr, patch_type, colormat, colorfmat, maskmat.clone(), size, emiter);

        //show err results
        //cv::minMaxLoc(nnferr, &errmin, &errmax);
        //std::cout << "Max error: " << errmax << std::endl;
        //cv::imshow("nnf error", nnferr/errmax);
        //cv::waitKey();

        //update a color image
        colorVoteLap(nnf, nnferr, patch_type, colormat,  colorfmat, maskmat.clone(), size);
    }

    //	free(patch_type);
    free(outputfilename);
}
