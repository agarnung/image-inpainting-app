#include "CahnHilliardImageInpainting.h"

#include <opencv4/opencv2/core.hpp>

CahnHilliardImageInpainting::CahnHilliardImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void CahnHilliardImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();

    // Reemplazamos los valores con los que proporcionaste
    mParameterSet->addParameter(QString("iters"),
                                5000,
                                QString("Iterations number"),
                                QString("Total number of iterations for the algorithm"),
                                true, 1, 5000);

    mParameterSet->addParameter(QString("D"),
                                2.5,
                                QString("Diffusion coefficient"),
                                QString("Diffusion coefficient D for the algorithm"),
                                true, 0.0, 10.0);

    mParameterSet->addParameter(QString("gamma"),
                                1.85,
                                QString("Reaction coefficient"),
                                QString("Weight of the reaction term"),
                                true, 0.0, 3.0);

    mParameterSet->addParameter(QString("dt"),
                                0.01,
                                QString("Time step"),
                                QString("Time step for the numerical scheme"),
                                true, 0.0001, 10.0);

    mParameterSet->addParameter(QString("deltaX"),
                                1.0,
                                QString("Spatial step in X direction"),
                                QString("Spacing in the X direction for numerical grid"),
                                true, 0.1, 10.0);

    mParameterSet->addParameter(QString("deltaY"),
                                1.0,
                                QString("Spatial step in Y direction"),
                                QString("Spacing in the Y direction for numerical grid"),
                                true, 0.1, 10.0);

    mParameterSet->addParameter(QString("useExplicitLaplacian"),
                                true,
                                QString("Use explicit Laplacian"),
                                QString("Use explicit Laplacian for the numerical method"));

    mParameterSet->setName(QString("Cahn-Hilliard Image Inpainting Algorithm"));
    mParameterSet->setLabel(QString("Cahn-Hilliard image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Cahn-Hilliard image inpainting algorithm -- Parameters"));
}

void CahnHilliardImageInpainting::cahnHilliardInpainting(cv::Mat& c, const cv::Mat& mask, int nIters, double D, double gamma, double dt, double deltaX, double deltaY, bool useExplicitLaplacian)
{
    cv::Mat c_new = c.clone();
    int height = c.rows;
    int width = c.cols;

    auto calcularMu = [](int r, int col, const double* c_ptr, int width, int height, double gamma, double& mu)
    {
        double d2x = 0.0, d2y = 0.0, laplacian_c = 0.0;

        const double c_ij = c_ptr[r * width + col];

        if (col != 0 && col != width - 1)
            d2x = c_ptr[r * width + (col + 1)] - 2 * c_ij + c_ptr[r * width + (col - 1)];
        if (r != 0 && r != height - 1)
            d2y = c_ptr[(r + 1) * width + col] - 2 * c_ij + c_ptr[(r - 1) * width + col];
        laplacian_c = d2x + d2y;

        mu = std::pow(c_ij, 3) - c_ij - gamma * laplacian_c;
    };

    for (int iter = 0; iter < nIters; iter++)
    {
        const double* c_ptr = c.ptr<double>();
        double* c_new_ptr = c_new.ptr<double>();

        for (int r = 0; r < height; ++r)
        {
            for (int col = 0; col < width; ++col)
            {
                // Imponer igualdad de imagen restaurada y original en las zonas no perdidas, i.e. distintas de cero
                if (mask.at<double>(r, col) != 0.0)
                {
                    c_new_ptr[r * width + col] = c_ptr[r * width + col];
                    continue;
                }

                double laplacian_mu = 0.0;
                double mu_cij = 0.0, mu_cim1j = 0.0, mu_cip1j = 0.0, mu_cijm1 = 0.0, mu_cijp1 = 0.0;

                const double c_ij = c_ptr[r * width + col];

                // Si no se usa el Laplaciano explícito
                if (!useExplicitLaplacian)
                {
                    if (col != 0 && col != width - 1) // x
                    {
                        calcularMu(r, col, c_ptr, width, height, gamma, mu_cij);
                        calcularMu(r, col - 1, c_ptr, width, height, gamma, mu_cijm1);
                        calcularMu(r, col + 1, c_ptr, width, height, gamma, mu_cijp1);
                        laplacian_mu += (mu_cijm1 - 2 * mu_cij + mu_cijp1) / std::pow(deltaX, 2);
                    }
                    if (r != 0 && r != height - 1) // y
                    {
                        calcularMu(r, col, c_ptr, width, height, gamma, mu_cij);
                        calcularMu(r - 1, col, c_ptr, width, height, gamma, mu_cim1j);
                        calcularMu(r + 1, col, c_ptr, width, height, gamma, mu_cip1j);
                        laplacian_mu += (mu_cim1j - 2 * mu_cij + mu_cip1j) / std::pow(deltaY, 2);
                    }
                }
                else
                {
                    // Cálculos más complejos del Laplaciano y las derivadas de cuarto orden
                    double d2x = 0.0, d2y = 0.0;
                    double d4x = 0.0, d4y = 0.0;
                    double laplacian_c = 0.0;

                    if (col != 0 && col != width - 1)
                        d2x = c_ptr[r * width + (col + 1)] - 2 * c_ij + c_ptr[r * width + (col - 1)];
                    if (r != 0 && r != height - 1)
                        d2y = c_ptr[(r + 1) * width + col] - 2 * c_ij + c_ptr[(r - 1) * width + col];
                    laplacian_c = d2x + d2y;

                    if (col > 1 && col < width - 2)
                        d4x = c_ptr[r * width + (col + 2)] - 4 * c_ptr[r * width + (col + 1)] + 6 * c_ptr[r * width + col] - 4 * c_ptr[r * width + (col - 1)] + c_ptr[r * width + (col - 2)];
                    if (r > 1 && r < height - 2)
                        d4y = c_ptr[(r + 2) * width + col] - 4 * c_ptr[(r + 1) * width + col] + 6 * c_ptr[r * width + col] - 4 * c_ptr[(r - 1) * width + col] + c_ptr[(r - 2) * width + col];

                    laplacian_mu = 6 * c_ij * laplacian_c - laplacian_c - gamma * (d4x + 2 * d2x * d2y + d4y);
                }

                // Actualizar la intensidad, solo en las zonas perdidas
                double nuevaIntensidad = std::max(0.0, std::min(1.0, c_ij + D * laplacian_mu * dt));
                c_new_ptr[r * width + col] = nuevaIntensidad;
            }
        }

        cv::swap(c, c_new);

        emit sendImageProcess(c_new);
    }
}

void CahnHilliardImageInpainting::inpaint()
{
    cv::Mat image = mDataManager->getOriginalImage();
    if (image.empty())
    {
        qWarning() << "The image is empty";
        return;
    }

    cv::Mat mask = mDataManager->getMask();
    if (mask.empty())
    {
        qWarning() << "The mask is empty";
        return;
    }

    int iters;
    double D, gamma, dt, deltaX, deltaY;
    bool useExplicitLaplacian;
    if (!mParameterSet->getValue(QString("iters"), iters) ||
        !mParameterSet->getValue(QString("D"), D) ||
        !mParameterSet->getValue(QString("gamma"), gamma) ||
        !mParameterSet->getValue(QString("dt"), dt) ||
        !mParameterSet->getValue(QString("deltaX"), deltaX) ||
        !mParameterSet->getValue(QString("deltaY"), deltaY) ||
        !mParameterSet->getValue(QString("useExplicitLaplacian"), useExplicitLaplacian))
    {
        qWarning() << "Could not retrieve all parameters.";
        return;
    }

    if (image.channels() == 4)
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);

    if (mask.depth() == CV_8U)
        mask.convertTo(mask, CV_64F, 1.0 / 255.0);
    else if (mask.depth() == CV_32F)
        mask.convertTo(mask, CV_64F);

    if (image.depth() == CV_8U)
        image.convertTo(image, CV_64F, 1.0 / 255.0);
    else if (image.depth() == CV_32F)
        image.convertTo(image, CV_64F);

    if (image.channels() == 1)
        cv::multiply(image, mask, image);
    else
    {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (int i = 0; i < (int)channels.size(); ++i)
            cv::multiply(channels[i], mask, channels[i]);
        cv::merge(channels, image);
    }

    if (image.channels() == 3)
    {
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        for (int i = 0; i < (int)channels.size(); ++i)
            cahnHilliardInpainting(channels[i], mask, iters, D, gamma, dt, deltaX, deltaY, useExplicitLaplacian);

        cv::merge(channels, image);
    }
    else
        cahnHilliardInpainting(image, mask, iters, D, gamma, dt, deltaX, deltaY, useExplicitLaplacian);

    image.convertTo(image, CV_8U, 255.0);

    mDataManager->setInpaintedImage(image);
}
