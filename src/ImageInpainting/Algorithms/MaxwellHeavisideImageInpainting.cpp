#include "MaxwellHeavisideImageInpainting.h"

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#include <QDebug>

MaxwellHeavisideImageInpainting::MaxwellHeavisideImageInpainting(DataManager* dataManager, ParameterSet* parameterSet)
    : ImageInpaintingBase(dataManager, parameterSet)
{
    initParameters();
}

void MaxwellHeavisideImageInpainting::initParameters()
{
    mParameterSet->removeAllParameter();
    mParameterSet->addParameter(QString("iters"),
                                1750,
                                QString("Iterations number"),
                                QString("Total number of iterations for the algorithm"),
                                true, 1, 5000);
    mParameterSet->addParameter(QString("c_wave"),
                                1.0,
                                QString("Wave speed"),
                                QString("Propagation speed of the wave"),
                                true, 0.0, 3e8);
    mParameterSet->addParameter(QString("dt"),
                                0.01,
                                QString("Time step"),
                                QString("Time step for the numerical scheme"),
                                true, 0.0001, 10.0);
    mParameterSet->addParameter(QString("alpha"),
                                0.75,
                                QString("Laplacian coefficient"),
                                QString("Weight of the Laplacian term"),
                                true, 0.0, 1.0);
    mParameterSet->addParameter(QString("beta"),
                                0.1,
                                QString("Divergence coefficient"),
                                QString("Weight of the divergence term"),
                                true, 0.0, 1.0);
    mParameterSet->addParameter(QString("gamma"),
                                0.05,
                                QString("Reaction coefficient"),
                                QString("Weight of the reaction term"),
                                true, 0.0, 1.0);
    mParameterSet->addParameter(QString("epsilon_0"),
                                1.0,
                                QString("epsilon_0"),
                                QString("epsilon_0"),
                                true, 0.0, 1.0);
    mParameterSet->addParameter(QString("mu_0"),
                                1.0,
                                QString("mu_0"),
                                QString("mu_0"),
                                true, 0.0, 1.0);
    mParameterSet->addParameter(QString("useEulerMethod"),
                                false,
                                QString("useEulerMethod"),
                                QString("Use Euler method (true) or RK4"));
    mParameterSet->addParameter(QString("stationaryFields"),
                                false,
                                QString("stationaryFields"),
                                QString("Use stationary (true) or dynamic fields"));
    mParameterSet->setName(QString("Maxwell-Heaviside Image Inpainting Algorithm"));
    mParameterSet->setLabel(QString("Maxwell-Heaviside image inpainting algorithm"));
    mParameterSet->setIntroduction(QString("Parameters settings"));
}

void MaxwellHeavisideImageInpainting::maxwellHeavisidePDEInpainting(cv::Mat& u, const cv::Mat& mask, int nIters, double c_wave,
                                                                    double dt, double alpha, double beta, double gamma,
                                                                    bool useEquationOne, double epsilon_0, double mu_0,
                                                                    bool useEulerMethod, bool stationaryFields)
{
    cv::Mat original = u.clone();

    cv::Mat kernel_x = (cv::Mat_<double>(1, 3) << -1, 0, 1) / 2.0;
    cv::Mat kernel_y = (cv::Mat_<double>(3, 1) << -1, 0, 1) / 2.0;

    auto laplacian = [](const cv::Mat& src, cv::Mat& dst) {
        cv::Mat kernel = (cv::Mat_<double>(3, 3) << 0, 1, 0,
                          1, -4, 1,
                          0, 1, 0);
        cv::filter2D(src, dst, CV_64F, kernel);
    };

    auto divergence = [kernel_x, kernel_y](const cv::Mat& Fx, const cv::Mat& Fy, cv::Mat& div) {
        cv::Mat dFx, dFy;
        cv::filter2D(Fx, dFx, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(Fy, dFy, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        div = dFx + dFy;
    };

    auto computeElectricField = [kernel_x, kernel_y](const cv::Mat& u, cv::Mat& Ex, cv::Mat& Ey) {
        cv::filter2D(u, Ex, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(u, Ey, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    };

    auto computeMagneticField = [kernel_x, kernel_y](const cv::Mat& u, cv::Mat& H) {
        cv::Mat ux, uy, uxy, uyx;
        cv::filter2D(u, ux, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(u, uy, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(ux, uxy, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(uy, uyx, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        H = uyx - uxy;
    };

    auto computeMaxwellHeavisideEMField = [useEquationOne, epsilon_0, mu_0](const cv::Mat& Ex, const cv::Mat& Ey, const cv::Mat& H, cv::Mat& EM) {
        if (useEquationOne)
        {
            //            cv::sqrt(Ex.mul(Ex) + Ey.mul(Ey) + H.mul(H), EM);
            //            EM = Ex.mul(Ex).mul(Ex) + Ey.mul(Ey).mul(Ey) + H.mul(H).mul(H);
            EM = Ex + Ey + H;
        }
        else
        {
            //            cv::Mat E = Ex.mul(Ex) + Ey.mul(Ey);
            cv::Mat E = Ex.mul(Ex).mul(Ex) + Ey.mul(Ey).mul(Ey);
            EM = 0.5 * (epsilon_0 * E + mu_0 * H.mul(H));
        }
    };

    cv::Mat Ek, EkMas1, Exk, ExkMas1, Eyk, EykMas1;
    cv::Mat Hk, HkMas1;

    /// Condiciones iniciales
    computeElectricField(u, Exk, Eyk);
    cv::sqrt(Exk.mul(Exk) + Eyk.mul(Eyk), Ek);
    computeMagneticField(u, Hk);


    for (int iter = 0; iter < nIters; iter++)
    {
        cv::Mat Ex, Ey, H, Hkx, Hky, EkxHky, EkyHkx, laplacian_u, divergence_EH, reaction_term, campoElectroMagnetico;
        if (useEulerMethod)
        {
            if (alpha != 0.0) laplacian(u, laplacian_u);
            else laplacian_u = cv::Mat::zeros(u.size(), CV_64FC1);

            /// Computar campo eléctrico
            if (beta != 0 || gamma != 0)
            {
                computeElectricField(u, Ex, Ey);
                cv::Mat E;
                cv::sqrt(Ex.mul(Ex) + Ey.mul(Ey), E);
                Ek -= c_wave * dt * E;
                if (!stationaryFields)
                {
                    cv::filter2D(Ek, Exk, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                    cv::filter2D(Ek, Eyk, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                }
            }

            /// Computar campo magnético
            if (beta != 0.0)
            {
                computeMagneticField(u, H);
                if (!stationaryFields)
                    Hk += c_wave * dt * H;

                cv::filter2D(Hk, Hkx, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(Hk, Hky, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                EkxHky = Exk.mul(Hky);
                EkyHkx = Eyk.mul(Hkx);
                divergence(EkxHky, EkyHkx, divergence_EH);
            } else divergence_EH = cv::Mat::zeros(u.size(), CV_64FC1);

            if (gamma != 0.0)
            {
                computeMaxwellHeavisideEMField(Exk, Eyk, Hk, campoElectroMagnetico);
                reaction_term = gamma * campoElectroMagnetico;
            } else reaction_term = cv::Mat::zeros(u.size(), CV_64FC1);

            u += dt * (alpha * laplacian_u - beta * divergence_EH + reaction_term);
        }
        else
        {
            /// -> Paso 1 de RK4
            if (alpha != 0.0) laplacian(u, laplacian_u);
            else laplacian_u = cv::Mat::zeros(u.size(), CV_64FC1);
            /// Computar campo eléctrico
            if (beta != 0 || gamma != 0)
            {
                computeElectricField(u, Ex, Ey);
                cv::Mat E;
                cv::sqrt(Ex.mul(Ex) + Ey.mul(Ey), E);
                Ek -= c_wave * dt * E;
                if (!stationaryFields)
                {
                    cv::filter2D(Ek, Exk, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                    cv::filter2D(Ek, Eyk, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                }
            }
            /// Computar campo magnético
            if (beta != 0.0)
            {
                computeMagneticField(u, H);
                if (!stationaryFields)
                    Hk += c_wave * dt * H;

                cv::filter2D(Hk, Hkx, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(Hk, Hky, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                EkxHky = Exk.mul(Hky);
                EkyHkx = Eyk.mul(Hkx);
                divergence(EkxHky, EkyHkx, divergence_EH);
            } else divergence_EH = cv::Mat::zeros(u.size(), CV_64FC1);
            if (gamma != 0.0)
            {
                computeMaxwellHeavisideEMField(Exk, Eyk, Hk, campoElectroMagnetico);
                reaction_term = gamma * campoElectroMagnetico;
            } else reaction_term = cv::Mat::zeros(u.size(), CV_64FC1);
            cv::Mat k1 = alpha * laplacian_u - beta * divergence_EH + gamma * reaction_term;

            /// -> Paso 2 de RK4
            cv::Mat u_temp = u + 0.5 * dt * k1;
            if (alpha != 0.0) laplacian(u_temp, laplacian_u);
            else laplacian_u = cv::Mat::zeros(u.size(), CV_64FC1);
            if (beta != 0 || gamma != 0)
            {
                computeElectricField(u_temp, Ex, Ey);
                cv::Mat E;
                cv::sqrt(Ex.mul(Ex) + Ey.mul(Ey), E);
                Ek -= c_wave * dt * E;
                if (!stationaryFields)
                {
                    cv::filter2D(Ek, Exk, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                    cv::filter2D(Ek, Eyk, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                }
            }
            /// Computar campo magnético
            if (beta != 0.0)
            {
                computeMagneticField(u, H);
                if (!stationaryFields)
                    Hk += c_wave * dt * H;

                cv::filter2D(Hk, Hkx, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(Hk, Hky, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                EkxHky = Exk.mul(Hky);
                EkyHkx = Eyk.mul(Hkx);
                divergence(EkxHky, EkyHkx, divergence_EH);
            } else divergence_EH = cv::Mat::zeros(u.size(), CV_64FC1);
            if (gamma != 0.0)
            {
                computeMaxwellHeavisideEMField(Exk, Eyk, Hk, campoElectroMagnetico);
                reaction_term = gamma * campoElectroMagnetico;
            } else reaction_term = cv::Mat::zeros(u.size(), CV_64FC1);
            cv::Mat k2 = alpha * laplacian_u - beta * divergence_EH + gamma * reaction_term;

            /// -> Paso 3 de RK4
            u_temp = u + 0.5 * dt * k2;
            if (alpha != 0.0) laplacian(u_temp, laplacian_u);
            else laplacian_u = cv::Mat::zeros(u.size(), CV_64FC1);
            /// Computar campo eléctrico
            if (beta != 0 || gamma != 0)
            {
                computeElectricField(u_temp, Ex, Ey);
                cv::Mat E;
                cv::sqrt(Ex.mul(Ex) + Ey.mul(Ey), E);
                Ek -= c_wave * dt * E;
                if (!stationaryFields)
                {
                    cv::filter2D(Ek, Exk, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                    cv::filter2D(Ek, Eyk, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                }
            }
            /// Computar campo magnético
            if (beta != 0.0)
            {
                computeMagneticField(u, H);
                if (!stationaryFields)
                    Hk += c_wave * dt * H;

                cv::filter2D(Hk, Hkx, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(Hk, Hky, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                EkxHky = Exk.mul(Hky);
                EkyHkx = Eyk.mul(Hkx);
                divergence(EkxHky, EkyHkx, divergence_EH);
            } else divergence_EH = cv::Mat::zeros(u.size(), CV_64FC1);

            if (gamma != 0.0)
            {
                computeMaxwellHeavisideEMField(Exk, Eyk, Hk, campoElectroMagnetico);
                reaction_term = gamma * campoElectroMagnetico;
            } else reaction_term = cv::Mat::zeros(u.size(), CV_64FC1);

            cv::Mat k3 = alpha * laplacian_u - beta * divergence_EH + gamma * reaction_term;

            /// -> Paso 4 de RK4
            u_temp = u + dt * k3;
            if (alpha != 0.0) laplacian(u_temp, laplacian_u);
            else laplacian_u = cv::Mat::zeros(u.size(), CV_64FC1);
            /// Computar campo eléctrico
            if (beta != 0 || gamma != 0)
            {
                computeElectricField(u_temp, Ex, Ey);
                cv::Mat E;
                cv::sqrt(Ex.mul(Ex) + Ey.mul(Ey), E);
                Ek -= c_wave * dt * E;
                if (!stationaryFields)
                {
                    cv::filter2D(Ek, Exk, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                    cv::filter2D(Ek, Eyk, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                }
            }
            /// Computar campo magnético
            if (beta != 0.0)
            {
                computeMagneticField(u, H);
                if (!stationaryFields)
                    Hk += c_wave * dt * H;

                cv::filter2D(Hk, Hkx, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::filter2D(Hk, Hky, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                EkxHky = Exk.mul(Hky);
                EkyHkx = Eyk.mul(Hkx);
                divergence(EkxHky, EkyHkx, divergence_EH);
            } else divergence_EH = cv::Mat::zeros(u.size(), CV_64FC1);

            if (gamma != 0.0)
            {
                computeMaxwellHeavisideEMField(Exk, Eyk, Hk, campoElectroMagnetico);
                reaction_term = gamma * campoElectroMagnetico;
            } else reaction_term = cv::Mat::zeros(u.size(), CV_64FC1);
            cv::Mat k4 = alpha * laplacian_u - beta * divergence_EH + gamma * reaction_term;

            /// Combinar las 4 contribuciones RK4
            cv::Mat delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
            u = u + dt * delta;
        }

        original.copyTo(u, mask != 0.0);

        emit sendImageProcess(u);
    }
}

void MaxwellHeavisideImageInpainting::inpaint()
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
    double c_wave, dt, alpha, beta, gamma, epsilon_0, mu_0;
    bool useEulerMethod, stationaryFields;
    if (!mParameterSet->getValue(QString("iters"), iters) ||
        !mParameterSet->getValue(QString("c_wave"), c_wave) ||
        !mParameterSet->getValue(QString("dt"), dt) ||
        !mParameterSet->getValue(QString("alpha"), alpha) ||
        !mParameterSet->getValue(QString("beta"), beta) ||
        !mParameterSet->getValue(QString("gamma"), gamma) ||
        !mParameterSet->getValue(QString("epsilon_0"), epsilon_0) ||
        !mParameterSet->getValue(QString("mu_0"), mu_0) ||
        !mParameterSet->getValue(QString("useEulerMethod"), useEulerMethod) ||
        !mParameterSet->getValue(QString("stationaryFields"), stationaryFields))
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
        {
            emit sendOtherMessage("Processing channel " + QString::number(i) + "...");
            maxwellHeavisidePDEInpainting(channels[i], mask, iters, c_wave, dt, alpha, beta, gamma, useEulerMethod, epsilon_0, mu_0, stationaryFields);
        }
        emit sendOtherMessage("");

        cv::merge(channels, image);
    }
    else
    {
        emit sendOtherMessage("Processing image...");
        maxwellHeavisidePDEInpainting(image, mask, iters, c_wave, dt, alpha, beta, gamma, useEulerMethod, epsilon_0, mu_0, stationaryFields);
        emit sendOtherMessage("");
    }

    image.convertTo(image, CV_8U, 255.0);

    mDataManager->setImage(image);
    mDataManager->setInpaintedImage(image);

    emit sendImageProcess(image);
}
