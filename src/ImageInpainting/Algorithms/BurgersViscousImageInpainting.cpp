#include "BurgersViscousImageInpainting.h"

BurgersViscousImageInpainting::BurgersViscousImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void BurgersViscousImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("iters"),
                                5,
                                QString("Iterations number"),
                                QString("The total number of iteration of the algorithm"),
                                true, 1, 100);
    mParameterSet->addParameter(QString("dx"),
                                0.01,
                                QString("dx"),
                                QString("Space step in x"),
                                true, 0.0001, 100.0);
    mParameterSet->addParameter(QString("dy"),
                                0.01,
                                QString("dy"),
                                QString("Space step in y"),
                                true, 0.0001, 100.0);
    mParameterSet->addParameter(QString("nu"),
                                0.001,
                                QString("nu"),
                                QString("Viscosity"),
                                true, 0.0001, 100.0);
    mParameterSet->addParameter(QString("dt"),
                                0.001,
                                QString("dt"),
                                QString("Time step"),
                                true, 0.0001, 100.0);
    mParameterSet->addParameter(QString("doUpwindDifferences"),
                                true,
                                QString("doUpwindDifferences"),
                                QString("doUpwindDifferences"));
    mParameterSet->setName(QString("Maxwell-Heaviside image inpainting algorithm"));
    mParameterSet->setLabel(QString("Maxwell-Heaviside image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Maxwell-Heaviside image inpainting algorithm -- Parameters"));
}

void BurgersViscousImageInpainting::burgersViscousEquationInpainting(cv::Mat& u, const cv::Mat& mask, double nu, int nIters, double dt, double dx, double dy, bool useUpwind)
{
    cv::Mat original = u.clone();

    int height = u.rows;
    int width = u.cols;

    cv::Mat uNew = u.clone();

    for (int iter = 0; iter < nIters; ++iter) {
        double* currentPtr = u.ptr<double>();
        double* nextPtr = uNew.ptr<double>();

        // Iteración para puntos interiores
        for (int r = 1; r < height - 1; ++r) {
            for (int c = 1; c < width - 1; ++c) {
                double u_ij = currentPtr[r * width + c];
                double u_left = currentPtr[r * width + (c - 1)];
                double u_right = currentPtr[r * width + (c + 1)];
                double u_up = currentPtr[(r - 1) * width + c];
                double u_down = currentPtr[(r + 1) * width + c];

                double du_dx, du_dy;

                if (useUpwind) {
                    // Derivadas espaciales usando esquema upwind
                    du_dx = (u_ij > 0)
                                ? (u_ij - u_left) / dx  // Flujo positivo: usa puntos "hacia atrás"
                                : (u_right - u_ij) / dx; // Flujo negativo: usa puntos "hacia adelante"

                    du_dy = (u_ij > 0)
                                ? (u_ij - u_down) / dy  // Flujo positivo: usa puntos "hacia atrás"
                                : (u_up - u_ij) / dy;   // Flujo negativo: usa puntos "hacia adelante"
                } else {
                    // Derivadas espaciales usando esquema de diferencias centradas
                    du_dx = (u_right - u_left) / (2 * dx);
                    du_dy = (u_up - u_down) / (2 * dy);
                }

                // Derivadas de segundo orden (difusión)
                double d2u_dx2 = (u_right - 2 * u_ij + u_left) / (dx * dx);
                double d2u_dy2 = (u_up - 2 * u_ij + u_down) / (dy * dy);

                // Términos no lineales y difusión
                double nonLinearTermX = u_ij * du_dx;
                double nonLinearTermY = u_ij * du_dy;

                nextPtr[r * width + c] = u_ij
                                         - dt * (nonLinearTermX + nonLinearTermY)  // Advección
                                         + nu * dt * (d2u_dx2 + d2u_dy2);          // Difusión
            }
        }

        // Condiciones de frontera (Neumann: gradiente nulo)
        for (int c = 0; c < width; ++c) {
            nextPtr[0 * width + c] = nextPtr[1 * width + c];               // Borde superior
            nextPtr[(height - 1) * width + c] = nextPtr[(height - 2) * width + c]; // Borde inferior
        }
        for (int r = 0; r < height; ++r) {
            nextPtr[r * width + 0] = nextPtr[r * width + 1];               // Borde izquierdo
            nextPtr[r * width + (width - 1)] = nextPtr[r * width + (width - 2)]; // Borde derecho
        }

        original.copyTo(uNew, mask != 0.0);

        cv::swap(u, uNew);

        emit sendImageProcess(uNew);
    }
}

void BurgersViscousImageInpainting::inpaint()
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
    double dx, dy, nu, dt;
    bool doUpwindDifferences;
    if (!mParameterSet->getValue(QString("iters"), iters) ||
        !mParameterSet->getValue(QString("dx"), dx) ||
        !mParameterSet->getValue(QString("dy"), dy) ||
        !mParameterSet->getValue(QString("nu"), nu) ||
        !mParameterSet->getValue(QString("dt"), dt) ||
        !mParameterSet->getValue(QString("doUpwindDifferences"), doUpwindDifferences))
    {
        qWarning() << "Could not retrieved all parameters.";
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
            burgersViscousEquationInpainting(channels[i], mask, nu, iters, dt, dx, dy, doUpwindDifferences);

        cv::merge(channels, image);
    }
    else
        burgersViscousEquationInpainting(image, mask, nu, iters, dt, dx, dy, doUpwindDifferences);

    image.convertTo(image, CV_8U, 255.0);

    mDataManager->setInpaintedImage(image);
}
