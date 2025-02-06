#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include <cvlibrary.h>

void fourierHeatEquationPDE(cv::Mat& u, int nIters = 1000, double alpha = 0.1, double dt = 1.0, double variableDiffusivity = false)
{
    int height = u.rows;
    int width = u.cols;

    cv::Mat uNew = u.clone();

    for (int iter = 0; iter < nIters; iter++)
    {
        double* diffusedPtr = u.ptr<double>();
        double* diffusedNewPtr = uNew.ptr<double>();

        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                double d2x{0.0}, d2y{0.0};

                const double diffused_ij = diffusedPtr[r * width + c];

                /// Aproximar laplaciana mediante diferencias centrales y gradiente constante = 0.0 en el contorno para que
                /// no funcionen como fuentes ni sumideros de calor (divergencia es cero == condiciones de contorno de Neumann == sistema aislado)
                if (c != 0 && c != width - 1)
                    d2x = diffusedPtr[r * width + (c + 1)] - 2 * diffused_ij + diffusedPtr[r * width + (c - 1)];

                if (r != 0 && r != height - 1)
                    d2y = diffusedPtr[(r + 1) * width + c] - 2 * diffused_ij + diffusedPtr[(r - 1) * width + c];

                double alpha_u;
                if (variableDiffusivity)
                    alpha_u = alpha * std::sqrt(d2x * d2x + d2y * d2y);
                else
                    alpha_u = alpha;

                double nuevaIntensidad = diffused_ij + alpha_u * (d2x + d2y) * dt;
                diffusedNewPtr[r * width + c] = nuevaIntensidad;
            }
        }

        cv::swap(u, uNew);

//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("fourierHeatEquationPDE", u);
//        cv::waitKey(1);
    }
}

void laplaceEquation(cv::Mat& u, int nIters = 1000, bool variableDiffusivity = false, double alpha = 0.1)
{
    int height = u.rows;
    int width = u.cols;

    cv::Mat uNew = u.clone();

    for (int iter = 0; iter < nIters; iter++)
    {
        double* currentPtr = u.ptr<double>();
        double* nextPtr = uNew.ptr<double>();

        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                double d2x{0.0}, d2y{0.0};

                const double currentValue = currentPtr[r * width + c];

                // Condiciones de contorno de Neumann: gradiente cero
                if (c != 0 && c != width - 1)
                    d2x = currentPtr[r * width + (c + 1)] - 2 * currentValue + currentPtr[r * width + (c - 1)];

                if (r != 0 && r != height - 1)
                    d2y = currentPtr[(r + 1) * width + c] - 2 * currentValue + currentPtr[(r - 1) * width + c];

                double alpha_u = variableDiffusivity
                                     ? alpha * std::sqrt(d2x * d2x + d2y * d2y)
                                     : alpha;

                // Resolver Laplaciana sin término de evolución temporal
                nextPtr[r * width + c] = currentValue + alpha_u * (d2x + d2y);
            }
        }

        cv::swap(u, uNew);

//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("laplaceEquation", u);
//        cv::waitKey(10);
    }
}

void peronaMalikPDE(cv::Mat& u, int nIters = 300, double lambda = 0.1, double sigma = 0.1, double kappa = 0.015, bool cUseEquation1 = false, bool useDiscretizationScheme1 = false)
{
    cv::Mat new_diffused = u.clone();

    int height = u.rows;
    int width = u.cols;

    for (int iter = 0; iter < nIters; iter++)
    {
        cv::Mat smoothDiffused;
        if (sigma > 0.0)
            cv::GaussianBlur(u, smoothDiffused, cv::Size(0, 0), sigma, sigma, cv::BORDER_REFLECT);

        double* diffusedPtr = u.ptr<double>();
        double* newDiffusedPtr = new_diffused.ptr<double>();
        double* smoothDiffusedPtr = smoothDiffused.ptr<double>();

        for (int r = 1; r < height - 1; r++)
        {
            for (int c = 1; c < width - 1; c++)
            {
                double cN, cS, cE, cW;
                double deltaN, deltaS, deltaE, deltaW;

                double dij = diffusedPtr[r * width + c];
                const double dN = diffusedPtr[r * width + (c - 1)];
                const double dS = diffusedPtr[r * width + (c + 1)];
                const double dE = diffusedPtr[(r + 1) * width + c];
                const double dW = diffusedPtr[(r - 1) * width + c];

                if (sigma > 0.0)
                {
                    double dGIij = smoothDiffusedPtr[r * width + c];
                    const double dGIN = smoothDiffusedPtr[r * width + (c - 1)];
                    const double dGIS = smoothDiffusedPtr[r * width + (c + 1)];
                    const double dGIE = smoothDiffusedPtr[(r + 1) * width + c];
                    const double dGIW = smoothDiffusedPtr[(r - 1) * width + c];

                    deltaN = dGIN - dGIij;
                    deltaS = dGIS - dGIij;
                    deltaE = dGIE - dGIij;
                    deltaW = dGIW - dGIij;
                }
                else
                {
                    deltaN = dN - dij;
                    deltaS = dS - dij;
                    deltaE = dE - dij;
                    deltaW = dW - dij;
                }

                /// Cálculo de la difusividad con condiciones de contorno de Neumann == sistema aislado
                double c_k;
                if (c != 0 && c != width - 1 && r != 0 && r != height - 1)
                {
                    if (cUseEquation1)
                    {
                        cN = std::exp(-1.0 * std::pow(std::fabs(deltaN) / kappa, 2));
                        cS = std::exp(-1.0 * std::pow(std::fabs(deltaS) / kappa, 2));
                        cE = std::exp(-1.0 * std::pow(std::fabs(deltaE) / kappa, 2));
                        cW = std::exp(-1.0 * std::pow(std::fabs(deltaW) / kappa, 2));
                    }
                    else
                    {
                        cN = 1.0 / (1 + std::pow(std::fabs(deltaN) / kappa, 2));
                        cS = 1.0 / (1 + std::pow(std::fabs(deltaS) / kappa, 2));
                        cE = 1.0 / (1 + std::pow(std::fabs(deltaE) / kappa, 2));
                        cW = 1.0 / (1 + std::pow(std::fabs(deltaW) / kappa, 2));
                    }

                    c_k = cN + cS + cE + cW;
                }
                else
                    c_k = 0.0;

                if (useDiscretizationScheme1)
                {
                    const double c_dot_nablaI = (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW);
                    newDiffusedPtr[r * width + c] = dij + lambda * c_dot_nablaI;
                }
                else
                {
                    const double c_dot_I = (cN * dN + cS * dS + cE * dE + cW * dW);
                    newDiffusedPtr[r * width + c] = dij * (1 - lambda * c_k) + lambda * c_dot_I;
                }
            }
        }

        cv::swap(u, new_diffused);

//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("matricialPeronaMalikPDE", u);
//        cv::waitKey(100);
//        showHeightmap3D(scatter, cv::Mat_<double>(u), 100.0f);
    }
}

void maxwellHeavisidePDE(cv::Mat& u, int nIters = 1000, double c_wave = 3.0e8, double dt = 0.1, double alpha = 0.0, double beta = 0.0, double gamma = 1.0, bool useEquationOne = true, double epsilon_0 = 1.0, double mu_0 = 1.0, bool useEulerMethod = true, bool stationaryFields = false)
{
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

//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("maxwellHea/visidePDE", u);
//        cv::waitKey(1);
//        showHeightmap3D(scatter, cv::Mat_<double>(u), 100.0f);
    }
}

void kdvEquation2D(cv::Mat& u, double alpha, double dt, int nIters, bool useUpwind) {
    int height = u.rows;  // Número de filas
    int width = u.cols;   // Número de columnas
    double dx = 1.0;      // Espaciado en x
    double dy = 1.0;      // Espaciado en y

    cv::Mat uNew = u.clone(); // Copia para actualizar valores

    for (int iter = 0; iter < nIters; ++iter) {
        double* currentU = u.ptr<double>();
        double* nextU = uNew.ptr<double>();

        for (int r = 2; r < height - 2; ++r) {
            for (int c = 2; c < width - 2; ++c) {
                double u_ij = currentU[r * width + c];

                // Valores vecinos
                double u_left = currentU[r * width + (c - 1)];
                double u_right = currentU[r * width + (c + 1)];
                double u_farLeft = currentU[r * width + (c - 2)];
                double u_farRight = currentU[r * width + (c + 2)];

                double u_up = currentU[(r - 1) * width + c];
                double u_down = currentU[(r + 1) * width + c];
                double u_farUp = currentU[(r - 2) * width + c];
                double u_farDown = currentU[(r + 2) * width + c];

                // Derivadas de primer orden (upwind o centradas)
                double du_dx, du_dy;
                if (useUpwind) {
                    // Esquema Upwind
                    du_dx = (u_ij > 0) ? (u_ij - u_left) / dx : (u_right - u_ij) / dx;
                    du_dy = (u_ij > 0) ? (u_ij - u_up) / dy : (u_down - u_ij) / dy;
                } else {
                    // Esquema Centradas
                    du_dx = (u_right - u_left) / (2 * dx);
                    du_dy = (u_down - u_up) / (2 * dy);
                }

                // Derivadas de tercer orden (centradas en x y y)
                double d3u_dx3 = (u_farRight - 2 * u_right + 2 * u_left - u_farLeft) / (2 * dx * dx * dx);
                double d3u_dy3 = (u_farDown - 2 * u_down + 2 * u_up - u_farUp) / (2 * dy * dy * dy);

                nextU[r * width + c] = u_ij
                    - dt * (u_ij * du_dx + u_ij * du_dy) // Términos no lineales
                    - alpha * dt * (d3u_dx3 + d3u_dy3);  // Términos de dispersión
            }
        }

        // Condiciones de frontera (Neumann: gradiente nulo)
        for (int c = 0; c < width; ++c) {
            nextU[0 * width + c] = nextU[1 * width + c];                 // Borde superior
            nextU[(height - 1) * width + c] = nextU[(height - 2) * width + c]; // Borde inferior
        }
        for (int r = 0; r < height; ++r) {
            nextU[r * width + 0] = nextU[r * width + 1];                 // Borde izquierdo
            nextU[r * width + (width - 1)] = nextU[r * width + (width - 2)]; // Borde derecho
        }

        // Intercambiar buffers
        cv::swap(u, uNew);

        // Mostrar resultados en cada iteración
//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("Korteweg-de Vries 2D", u);
//        cv::waitKey(1);
    }
}

void kuramotoSivashinsky(cv::Mat& u, int nIters = 1000, double dt = 0.01, bool useUpwind = true) {
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

                // Derivadas espaciales
                double du_dx, du_dy;

                if (useUpwind) {
                    // Derivadas usando el esquema upwind
                    du_dx = (u_ij > 0)
                        ? (u_ij - u_left)  // Flujo positivo: usa puntos "hacia atrás"
                        : (u_right - u_ij); // Flujo negativo: usa puntos "hacia adelante"

                    du_dy = (u_ij > 0)
                        ? (u_ij - u_down)  // Flujo positivo: usa puntos "hacia atrás"
                        : (u_up - u_ij);   // Flujo negativo: usa puntos "hacia adelante"
                } else {
                    // Derivadas usando el esquema de diferencias centradas
                    du_dx = (u_right - u_left) / 2.0;
                    du_dy = (u_up - u_down) / 2.0;
                }

                // Derivadas segunda y cuarta en x y y
                double d2u_dx2 = (u_right - 2 * u_ij + u_left); // Segunda derivada en x
                double d2u_dy2 = (u_up - 2 * u_ij + u_down);   // Segunda derivada en y

                // Cálculo de la cuarta derivada en x
                double d4u_dx4 = 0.0;
                if (c >= 2 && c < width - 2) {
                    d4u_dx4 = (currentPtr[r * width + (c + 2)] - 4 * currentPtr[r * width + (c + 1)] + 6 * u_ij
                                - 4 * currentPtr[r * width + (c - 1)] + currentPtr[r * width + (c - 2)]); // Cuarta derivada en x
                }

                // Cálculo de la cuarta derivada en y
                double d4u_dy4 = 0.0;
                if (r >= 2 && r < height - 2) {
                    d4u_dy4 = (currentPtr[(r + 2) * width + c] - 4 * currentPtr[(r + 1) * width + c] + 6 * u_ij
                                - 4 * currentPtr[(r - 1) * width + c] + currentPtr[(r - 2) * width + c]); // Cuarta derivada en y
                }

                // Término no lineal
                double nonLinearTermX = u_ij * du_dx;
                double nonLinearTermY = u_ij * du_dy;

                // Ecuación de Kuramoto-Sivashinsky en 2D
                nextPtr[r * width + c] = u_ij
                    - dt * (d4u_dx4 + d2u_dx2 + d4u_dy4 + d2u_dy2 + nonLinearTermX + nonLinearTermY);  // Actualización de la variable u
            }
        }

        // Condiciones de frontera (Neumann: gradiente nulo)
        // Borde superior e inferior
        for (int c = 0; c < width; ++c) {
            nextPtr[0 * width + c] = nextPtr[1 * width + c];               // Borde superior
            nextPtr[(height - 1) * width + c] = nextPtr[(height - 2) * width + c]; // Borde inferior
        }

        // Borde izquierdo y derecho
        for (int r = 0; r < height; ++r) {
            nextPtr[r * width + 0] = nextPtr[r * width + 1];               // Borde izquierdo
            nextPtr[r * width + (width - 1)] = nextPtr[r * width + (width - 2)]; // Borde derecho
        }

        cv::swap(u, uNew);

//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("Kuramoto-Sivashinsky Equation 2D", u);
//        cv::waitKey(1);
    }
}

cv::Mat deconvolutionByTV(const cv::Mat& inputImage, const cv::Mat& kernel, int iterations = 100, double lambda = 0.1, double epsilon = 0.004) {
    cv::Mat fTV;
    inputImage.copyTo(fTV);

    int width = inputImage.cols;
    int height = inputImage.rows;

    cv::Mat gradientX = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat gradientY = cv::Mat::zeros(height, width, CV_64F);

    int padSize = kernel.rows / 2;
    cv::Mat kernelPadded;
    cv::copyMakeBorder(kernel, kernelPadded, padSize, padSize, padSize, padSize, cv::BORDER_CONSTANT, cv::Scalar(0));

    for (int niter = 0; niter < iterations; niter++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double curValue = fTV.at<double>(y, x);
                if (y < height - 1) {
                    gradientY.at<double>(y, x) = fTV.at<double>(y + 1, x) - curValue; // Gradiente en Y
                }
                if (x < width - 1) {
                    gradientX.at<double>(y, x) = fTV.at<double>(y, x + 1) - curValue; // Gradiente en X
                }
            }
        }

        cv::Mat normGrad;
        cv::magnitude(gradientX, gradientY, normGrad);
        normGrad += epsilon;

        gradientX = gradientX / normGrad;
        gradientY = gradientY / normGrad;

        cv::Mat divergence = cv::Mat::zeros(height, width, CV_64F);
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                divergence.at<double>(y, x) = gradientX.at<double>(y, x) - gradientX.at<double>(y - 1, x) +
                                              gradientY.at<double>(y, x) - gradientY.at<double>(y, x - 1);
            }
        }

        cv::Mat previous_fTV = fTV.clone();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double kValue = (1.0 / (1.0 + lambda * normGrad.at<double>(y, x)));

                double fidelityTerm = fTV.at<double>(y, x) - inputImage.at<double>(y, x);

                fTV.at<double>(y, x) -= (0.05 * kValue * (divergence.at<double>(y, x) + lambda * fidelityTerm));

                fTV.at<double>(y, x) = std::max(0.0, std::min(1.0, fTV.at<double>(y, x)));
            }
        }

//        cv::imshow("fTV", fTV);
//        cv::waitKey(5);
    }

    return fTV;
}

cv::Mat deconvolutionByTikhonov(const cv::Mat& inputImage, const cv::Mat& kernel, int iterations = 100, double lambda = 0.1) {
    cv::Mat fTikhonov;
    inputImage.convertTo(fTikhonov, CV_64F); // Asegurarse de que es de tipo doble

    cv::Mat kernelFloat;
    kernel.convertTo(kernelFloat, CV_64F);

    int padSizeX = kernel.cols / 2;
    int padSizeY = kernel.rows / 2;

    cv::Mat kernelPadded;
    cv::copyMakeBorder(kernelFloat, kernelPadded, padSizeY, padSizeY, padSizeX, padSizeX, cv::BORDER_CONSTANT, cv::Scalar(0));

    for (int niter = 0; niter < iterations; niter++) {
        cv::Mat convolvedImage;
        cv::filter2D(fTikhonov, convolvedImage, CV_64F, kernelPadded, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        cv::Mat fidelityTerm = convolvedImage - inputImage;

        cv::Mat fidelityBackprop;
        cv::filter2D(fidelityTerm, fidelityBackprop, CV_64F, kernelPadded, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);

        cv::Mat updateStep = fTikhonov - 0.1 * (fidelityBackprop + lambda * fTikhonov);

        cv::min(cv::max(updateStep, 0), 1, fTikhonov);

//        cv::imshow("fTikhonov", fTikhonov);
//        cv::waitKey(1);
    }

    return fTikhonov;
}

void cahnHilliardInpainting(cv::Mat& c, const cv::Mat& mask, int nIters = 50, double D = 1.5, double gamma = 1.0, double dt = 0.01, double deltaX = 1.0, double deltaY = 1.0, bool useExplicitLaplacian = true)
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

//        std::cout << "Iteración " << iter << std::endl;
//        cv::imshow("Cahn-Hilliard Inpainting", c_new);
//        cv::waitKey(1);
    }
}

void maxwellHeavisidePDEInpainting(cv::Mat& u, const cv::Mat& mask, int nIters = 1000, double c_wave = 3.0e8, double dt = 0.1, double alpha = 0.0, double beta = 0.0, double gamma = 1.0, bool useEquationOne = true, double epsilon_0 = 1.0, double mu_0 = 1.0, bool useEulerMethod = true, bool stationaryFields = false)
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

//        std::cout << "Iteración " << iter << std::endl;
        cv::imshow("maxwellHeavisidePDE", u);
        cv::waitKey(1);
    }
}

void burgersViscousEquationInpainting(cv::Mat& u, const cv::Mat& mask, double nu, int nIters = 1000, double dt = 0.01, double dx = 0.1, double dy = 0.1, bool useUpwind = true)
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

        std::cout << "Iteración " << iter << std::endl;
        cv::imshow("Burgers Equation 2D", uNew);
        cv::waitKey(1);
    }
}

void cahnHilliardPDE(cv::Mat& c, int nIters = 50, double D = 1.5, double gamma = 1.0, double dt = 0.01, double deltaX = 1.0, double deltaY = 1.0, bool useExplicitLaplacian = true)
{
    cv::Mat c_new = c.clone();

    int height = c.rows;
    int width = c.cols;

    auto calcularMu = [](int r, int col, const double* c_ptr, int width, int height, double gamma, double& mu)
    {
        double d2x = 0.0, d2y = 0.0, laplacian_c = 0.0;

        const double c_ij = c_ptr[r * width + col];

        /// Laplaciano de c
        if (col != 0 && col != width - 1)
            d2x = c_ptr[r * width + (col + 1)] - 2 * c_ij + c_ptr[r * width + (col - 1)];
        if (r != 0 && r != height - 1)
            d2y = c_ptr[(r + 1) * width + col] - 2 * c_ij + c_ptr[(r - 1) * width + col];
        laplacian_c = d2x + d2y;

        /// Potencial químico μ
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
                double laplacian_mu = 0.0;
                double mu_cij = 0.0, mu_cim1j = 0.0, mu_cip1j = 0.0, mu_cijm1 = 0.0, mu_cijp1 = 0.0;

                const double c_ij = c_ptr[r * width + col];

                /// Laplaciano de μ
                if (!useExplicitLaplacian)
                {
                    /// ∇2μ = ∂x2∂2μ​+∂y2∂2μ​ ≈ ∂x2∂2μ​≈μi+1,j​−2μi,j​+μi−1,j + μi,j+1​−2μi,j​+μi,j−1
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
                    /// ∇2μ=2c∇2c+3(∇c)2−γ∇2(∇2c), ∇2(∇2c)=∂x4∂4c​+2∂x2∂y2∂4c​+∂y4∂4c​
                    /// @todo Usar otras aproximaciones más precisas https://math.stackexchange.com/questions/1380848/looking-for-finite-difference-approximations-past-the-fourth-derivative
                    double d2x = 0.0, d2y = 0.0;
                    double d4x = 0.0, d4y = 0.0;
                    double laplacian_c = 0.0;

                    /// Laplaciano de c
                    if (col != 0 && col != width - 1)
                        d2x = c_ptr[r * width + (col + 1)] - 2 * c_ij + c_ptr[r * width + (col - 1)];
                    if (r != 0 && r != height - 1)
                        d2y = c_ptr[(r + 1) * width + col] - 2 * c_ij + c_ptr[(r - 1) * width + col];
                    laplacian_c = d2x + d2y;
                    /// Derivadas parciales de cuarto orden para c
                    if (col > 1 && col < width - 2)
                        d4x = c_ptr[r * width + (col + 2)] - 4 * c_ptr[r * width + (col + 1)] + 6 * c_ptr[r * width + col] - 4 * c_ptr[r * width + (col - 1)] + c_ptr[r * width + (col - 2)];
                    if (r > 1 && r < height - 2)
                        d4y = c_ptr[(r + 2) * width + col] - 4 * c_ptr[(r + 1) * width + col] + 6 * c_ptr[r * width + col] - 4 * c_ptr[(r - 1) * width + col] + c_ptr[(r - 2) * width + col];

                    laplacian_mu = 6 * c_ij * laplacian_c -
                            laplacian_c -
                            gamma * (d4x + 2 * d2x * d2y + d4y);
                }

                double nuevaIntensidad = std::max(0.0, std::min(1.0, c_ij + D * laplacian_mu * dt));
                c_new_ptr[r * width + col] = nuevaIntensidad;
            }
        }

        cv::swap(c, c_new);

//        std::cout << "Iteración " << iter << std::endl;
        cv::imshow("Cahn-Hilliard", c_new);
        cv::waitKey(1);
    }
}

int main()
{
    /// Denoising
    std::cout << "\nDenoising:" << std::endl;
//    {
//        cv::Mat input = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/lena.png", CV_LOAD_IMAGE_UNCHANGED);
//        input.convertTo(input, CV_64FC1, 1.0 / 255.0, 0.0);

//        cv::Mat noisy = cvlib::addGaussianNoise(input, 0.25);
//        cv::imshow("input", input);
//        cv::imshow("noisy", noisy);

//        {
//            int iters = 50;
//            double alpha = 0.1;
//            double dt = 0.1;
//            bool variable_diffusivity = false;
//            cv::Mat denoised = noisy.clone();
//            fourierHeatEquationPDE(denoised, iters, alpha, dt, variable_diffusivity);
//            cv::imshow("fourierHeatEquationPDE", denoised);
//            std::cout << "Results heat equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, denoised) << "/" <<
//                         cvlib::PSNR(input, denoised) << "/" <<
//                         cvlib::SSIM(input, denoised) << std::endl;
//        }

//        {
//            int iters = 50;
//            double alpha = 0.1;
//            bool variableDiffusivity = false;
//            cv::Mat denoised = noisy.clone();
//            laplaceEquation(denoised, iters, variableDiffusivity, alpha);
//            cv::imshow("laplaceEquation", denoised);
//            std::cout << "Results Laplace equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, denoised) << "/" <<
//                         cvlib::PSNR(input, denoised) << "/" <<
//                         cvlib::SSIM(input, denoised) << std::endl;
//        }

//        {
//            int iters = 50;
//            double alpha = 0.1;
//            double K = 0.05;
//            cv::Mat noisy_8UC3 = noisy.clone();
//            noisy_8UC3.convertTo(noisy_8UC3, CV_8UC1, 255.0, 0.0);
//            cv::cvtColor(noisy_8UC3, noisy_8UC3, cv::COLOR_GRAY2BGR, 0);
//            cv::Mat denoised;
//            cv::ximgproc::anisotropicDiffusion(noisy_8UC3, denoised, alpha, K, iters);
//            cv::cvtColor(denoised, denoised, cv::COLOR_BGR2GRAY, 0);
//            denoised.convertTo(denoised, CV_64FC1, 1.0 / 255.0, 0.0);
//            cv::imshow("anisotropicDiffusion", denoised);
//            std::cout << "Results anisotropic diffusion (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, denoised) << "/" <<
//                         cvlib::PSNR(input, denoised) << "/" <<
//                         cvlib::SSIM(input, denoised) << std::endl;
//        }

//        {
//            int iters = 50;
//            double lambda = 0.1;
//            double sigma = 0.1;
//            double kappa = 0.015;
//            bool useEquation1 = true;
//            bool useDiscretizationScheme1 = true;
//            cv::Mat denoised = noisy.clone();
//            peronaMalikPDE(denoised, iters, lambda, sigma, kappa, useEquation1, useDiscretizationScheme1);
//            cv::imshow("peronaMalikPDE", denoised);
//            std::cout << "Results Perona-Malik (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, denoised) << "/" <<
//                         cvlib::PSNR(input, denoised) << "/" <<
//                         cvlib::SSIM(input, denoised) << std::endl;
//        }

//        {
//            int nIters = 100;
//            double c_wave = 1.0;
//            double dt = 0.01;
//            double alpha = 0.55;
//            double beta = 1.0;
//            double gamma = 0.1;
//            bool useEquationOne = true;
//            double epsilon_0 = 1.0;
//            double mu_0 = 1.0;
//            bool useEulerMethod = true;
//            bool stationaryFields = false;
//            cv::Mat denoised = noisy.clone();
//            maxwellHeavisidePDE(denoised, nIters, c_wave, dt, alpha, beta, gamma, useEquationOne, epsilon_0, mu_0, useEulerMethod, stationaryFields);
//            cv::imshow("maxwellHeavisidePDE", denoised);
//            std::cout << "Results Maxwell-Heaviside equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, denoised) << "/" <<
//                         cvlib::PSNR(input, denoised) << "/" <<
//                         cvlib::SSIM(input, denoised) << std::endl;
//        }

//        cv::waitKey(0);
//    }

    /// Deblurring
    std::cout << "\nDeblurring:" << std::endl;
//    {
//        cv::Mat input = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/kiel.pgm", CV_LOAD_IMAGE_UNCHANGED);
//        input.convertTo(input, CV_64FC1, 1.0 / 255.0, 0.0);
//        int crop = 5;
//        int postCrop = 10;
//        cv::Mat extended;
//        cv::copyMakeBorder(input, extended, crop, crop, crop, crop, cv::BORDER_WRAP);

//        cv::imshow("input", input);
//        cv::imshow("extended", extended);

//        int qi_size = 11;
//        int kernel_size = 15;
//        int sigma = 3.0;
//        cv::Mat blurry;
//        cv::GaussianBlur(extended, blurry, cv::Size(kernel_size, kernel_size), sigma, sigma, cv::BORDER_REPLICATE);
//        blurry = cvlib::addGaussianNoise(blurry, 0.05);

//        cv::Mat blurry_vis;
//        blurry_vis = blurry(cv::Rect(postCrop, postCrop, extended.cols - postCrop * 2, extended.rows - postCrop * 2)).clone();
//        cv::imshow("blurry", blurry_vis);

//        {
//            int iters = 100;
//            double lambda = 50;
//            double epsilon = 0.01;
//            cv::Mat psf = cv::getGaussianKernel(7, 3, CV_64F) * cv::getGaussianKernel(7, 3, CV_64F).t();
//            cv::Mat deblurred = deconvolutionByTV(blurry, psf, iters, lambda, epsilon);
//            deblurred = deblurred(cv::Rect(postCrop, postCrop, deblurred.cols - postCrop * 2, deblurred.rows - postCrop * 2)).clone();
//            input = extended(cv::Rect(postCrop, postCrop, extended.cols - postCrop * 2, extended.rows - postCrop * 2)).clone();
//            cv::imshow("deconvolutionByTV", deblurred);
//            std::cout << "Results TV-deconovlution (Q/PSNR): " << std::fixed << std::setprecision(4) <<
//                         cvlib::img_qi(input, deblurred, qi_size) << "/" <<
//                         cvlib::PSNR(input, deblurred) << std::endl;
//        }

//        {
//            int iters = 100;
//            double lambda = 0.15;
//            cv::Mat psf = cv::getGaussianKernel(7, 3, CV_64F) * cv::getGaussianKernel(7, 3, CV_64F).t();
//            cv::Mat deblurred = deconvolutionByTikhonov(blurry, psf, iters, lambda);
//            deblurred = deblurred(cv::Rect(postCrop, postCrop, deblurred.cols - postCrop * 2, deblurred.rows - postCrop * 2)).clone();
//            input = extended(cv::Rect(postCrop, postCrop, extended.cols - postCrop * 2, extended.rows - postCrop * 2)).clone();
//            cv::imshow("deconvolutionByTikhonov", deblurred);
//            std::cout << "Results Tikhonov-deconovlution (Q/PSNR): " << std::fixed << std::setprecision(4) <<
//                         cvlib::img_qi(input, deblurred, qi_size) << "/" <<
//                         cvlib::PSNR(input, deblurred) << std::endl;
//        }

//        {
//            const int iters = 1000;
//            const double dt = 0.0005;
//            const bool doUpwindDifferences = true;
//            cv::Mat deblurred = blurry.clone();
//            kuramotoSivashinsky(deblurred, iters, dt, doUpwindDifferences);
//            deblurred = deblurred(cv::Rect(postCrop, postCrop, deblurred.cols - postCrop * 2, deblurred.rows - postCrop * 2)).clone();
//            input = extended(cv::Rect(postCrop, postCrop, extended.cols - postCrop * 2, extended.rows - postCrop * 2)).clone();
//            cv::imshow("kuramotoSivashinsky", deblurred);
//            std::cout << "Results Kuramoto-Sivashinsky equation (Q/PSNR): " << std::fixed << std::setprecision(4) <<
//                         cvlib::img_qi(input, deblurred, qi_size) << "/" <<
//                         cvlib::PSNR(input, deblurred) << std::endl;
//        }

//        cv::waitKey(0);
//    }

    /// Enhancing
    std::cout << "\nEnhancing:" << std::endl;
    {
//        cv::Mat input = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/boat.tif", CV_LOAD_IMAGE_UNCHANGED);
//        cv::resize(input, input, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
//        input.convertTo(input, CV_64FC1, 1.0 / 255.0, 0.0);

//        cv::imshow("input", input);

//        std::cout << "Initial value: " << cvlib::computeAverageGradient(input) << std::endl;

////        {
////            cv::Mat input_8U, enhanced;
////            input.convertTo(input_8U, CV_8U,255.0, 0.0);
////            if (input_8U.channels() == 3)
////            {
////                cv::Mat yuv;
////                cv::cvtColor(input_8U, yuv, cv::COLOR_BGR2YUV);
////                std::vector<cv::Mat> yuv_channels(3);
////                cv::split(yuv, yuv_channels);
////                cv::Mat Y = yuv_channels[0];
////                cv::equalizeHist(Y, Y);
////                yuv_channels[0] = Y;
////                cv::merge(yuv_channels, yuv);
////                cv::cvtColor(yuv, enhanced, cv::COLOR_YUV2BGR);
////                enhanced.convertTo(enhanced, CV_64F, 1.0 / 255.0, 0.0);
////            }
////            else
////            {
////                cv::Mat enhanced;
////                cv::equalizeHist(input_8U, enhanced);
////                enhanced.convertTo(enhanced, CV_64FC1, 1.0 / 255.0, 0.0);
////            }

////            cv::imshow("equalizeHist", enhanced);
////            std::cout << "Result HE: " <<
////            cvlib::computeAverageGradient(enhanced) << "/" <<
////            cvlib::PSNR(input, enhanced) << std::endl;
////        }

////        {
////            double clip_limit = 1.0;
////            double grid_size = 8;
////            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_limit, cv::Size(grid_size, grid_size));

////            cv::Mat input_8U, enhanced;
////            input.convertTo(input_8U, CV_8U,255.0, 0.0);
////            if (input_8U.channels() == 3)
////            {
////                cv::Mat yuv;
////                cv::cvtColor(input_8U, yuv, cv::COLOR_BGR2YUV);
////                std::vector<cv::Mat> yuv_channels(3);
////                cv::split(yuv, yuv_channels);
////                cv::Mat Y = yuv_channels[0];
////                clahe->apply(Y, Y);
////                yuv_channels[0] = Y;
////                cv::merge(yuv_channels, yuv);
////                cv::cvtColor(yuv, enhanced, cv::COLOR_YUV2BGR);
////                enhanced.convertTo(enhanced, CV_64F, 1.0 / 255.0, 0.0);
////            }
////            else
////            {
////                cv::Mat enhanced;
////                clahe->apply(input_8U, enhanced);
////                enhanced.convertTo(enhanced, CV_64FC1, 1.0 / 255.0, 0.0);
////            }

////            cv::imshow("CLAHE", enhanced);
////            std::cout << "Result CLAHE: " <<
////            cvlib::computeAverageGradient(enhanced) << "/" <<
////            cvlib::PSNR(input, enhanced) << std::endl;
////        }

//        {
//            int nIters = 1000;
//            double D = -0.01;
//            double gamma = -0.05;
//            double dt = 0.01;
//            double deltaX = 1.0;
//            double deltaY = 1.0;
//            bool useExplicitLaplacian = true;
//            cv::Mat enhanced;
//            input.convertTo(enhanced, CV_32F);
//            if (enhanced.channels() == 3)
//            {
//                cv::Mat yuv;
//                cv::cvtColor(enhanced, yuv, cv::COLOR_BGR2YUV);
//                std::vector<cv::Mat> yuv_channels(3);
//                cv::split(yuv, yuv_channels);
//                cv::Mat Y = yuv_channels[0];
//                Y.convertTo(Y, CV_64F);
//                cahnHilliardPDE(Y, nIters, D, gamma, dt, deltaX, deltaY, useExplicitLaplacian);
//                Y.convertTo(Y, CV_32F);
//                yuv_channels[0] = Y;
//                cv::merge(yuv_channels, yuv);
//                cv::cvtColor(yuv, enhanced, cv::COLOR_YUV2BGR);
//            }
//            else
//            {
//                input.convertTo(enhanced, CV_64F);
//                cahnHilliardPDE(enhanced, nIters, D, gamma, dt, deltaX, deltaY, useExplicitLaplacian);
//            }

//            enhanced.convertTo(enhanced, CV_64F);
//            cv::imshow("Cahn-Hilliard", enhanced);
//            std::cout << "Result Cahn-Hilliard equation: " <<
//                         cvlib::computeAverageGradient(enhanced) << "/" <<
//                         cvlib::PSNR(input, enhanced) << std::endl;
//        }

//        {
//            int nIters = 500;
//            double c_wave = 2.5;
//            double dt = 0.01;
//            double alpha = -0.01;
//            double beta = 0.75;
//            double gamma = 0.5;
//            bool useEquationOne = false;
//            double epsilon_0 = 2.5;
//            double mu_0 = 5.5;
//            bool useEulerMethod = false;
//            bool stationaryFields = true;
//            cv::Mat enhanced;
//            if (enhanced.channels() == 3)
//            {
//                input.convertTo(enhanced, CV_32F);
//                cv::Mat yuv;
//                cv::cvtColor(enhanced, yuv, cv::COLOR_BGR2YUV);
//                std::vector<cv::Mat> yuv_channels(3);
//                cv::split(yuv, yuv_channels);
//                cv::Mat Y = yuv_channels[0];
//                Y.convertTo(Y, CV_64F);
//                maxwellHeavisidePDE(Y, nIters, c_wave, dt, alpha, beta, gamma, useEquationOne, epsilon_0, mu_0, useEulerMethod, stationaryFields);
//                Y.convertTo(Y, CV_32F);
//                yuv_channels[0] = Y;
//                cv::merge(yuv_channels, yuv);
//                cv::cvtColor(yuv, enhanced, cv::COLOR_YUV2BGR);
//            }
//            else
//            {
//                input.convertTo(enhanced, CV_64F);
//                maxwellHeavisidePDE(enhanced, nIters, c_wave, dt, alpha, beta, gamma, useEquationOne, epsilon_0, mu_0, useEulerMethod, stationaryFields);
//            }

//            enhanced.convertTo(enhanced, CV_64F);
//            cv::imshow("maxwellHeavisidePDE", enhanced);
//            std::cout << "Result Maxwell-Heaviside equation: " <<
//                         cvlib::computeAverageGradient(enhanced) << "/" <<
//                         cvlib::PSNR(input, enhanced) << std::endl;
//        }

//        cv::waitKey(0);
    }

    /// Inpainting
    std::cout << "\nInpainting:" << std::endl;
    {
//        cv::Mat input = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/camera.tif", CV_IMWRITE_PAM_FORMAT_GRAYSCALE);
//        cv::resize(input, input, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
//        input.convertTo(input, CV_64FC1, 1.0 / 255.0, 0.0);

////        input = cvlib::addGaussianNoise(input, 0.05);

//        cv::Mat mask(input.size(), CV_64FC1);
//        cvlib::createRandomInpaintingMask(0.25, mask);
//        cv::Mat corrupted;
//        cv::multiply(input, mask, corrupted);

//        cv::imshow("input", input);
//        cv::imshow("corrupted", corrupted);

//        {
//            cv::Mat corrupted_32FC1, mask_8UC1;
//            corrupted.convertTo(corrupted_32FC1, CV_32FC1);
//            mask.convertTo(mask_8UC1, CV_8UC1, 255.0);
//            mask_8UC1 = ~mask_8UC1;
//            cv::Mat inpainted;
//            cv::inpaint(corrupted_32FC1, mask_8UC1, inpainted, 3, cv::INPAINT_NS);
//            inpainted.convertTo(inpainted, CV_64FC1);
//            cv::imshow("inpainted INPAINT_NS", inpainted);
//            std::cout << "Results Navier-Stokes equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, inpainted) << "/" <<
//                         cvlib::PSNR(input, inpainted) << "/" <<
//                         cvlib::SSIM(input, inpainted) << std::endl;
//        }

//        {
//            cv::Mat corrupted_8UC1, mask_8UC1;
//            corrupted.convertTo(corrupted_8UC1, CV_8UC1, 255.0);
//            mask.convertTo(mask_8UC1, CV_8UC1, 255.0);
//            mask_8UC1 = ~mask_8UC1;
//            cv::Mat inpainted;
//            cv::inpaint(corrupted_8UC1, mask_8UC1, inpainted, 3, cv::INPAINT_TELEA);
//            inpainted.convertTo(inpainted, CV_64FC1, 1.0 / 255.0);
//            cv::imshow("inpainted INPAINT_TELEA", inpainted);
//            std::cout << "Results TELEA (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, inpainted) << "/" <<
//                         cvlib::PSNR(input, inpainted) << "/" <<
//                         cvlib::SSIM(input, inpainted) << std::endl;
//        }

//        {
//            int nIters = 5000;
//            double D = 2.5;
//            double gamma = 1.85;
//            double dt = 0.01;
//            double deltaX = 1.0;
//            double deltaY = 1.0;
//            bool useExplicitLaplacian = true;
//            cv::Mat inpainted = corrupted.clone();
//            cahnHilliardInpainting(inpainted, mask, nIters, D, gamma, dt, deltaX, deltaY, useExplicitLaplacian);
//            cv::imshow("cahnHilliardInpainting", inpainted);
//            std::cout << "Results Cahn–Hilliard equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, inpainted) << "/" <<
//                         cvlib::PSNR(input, inpainted) << "/" <<
//                         cvlib::SSIM(input, inpainted) << std::endl;
//        }

//        {
//            int nIters = 1750;
//            double c_wave = 1.0;
//            double dt = 0.01;
//            double alpha = 0.75;
//            double beta = 0.1;
//            double gamma = 0.05;
//            bool useEquationOne = true;
//            double epsilon_0 = 1.0;
//            double mu_0 = 1.0;
//            bool useEulerMethod = false;
//            bool stationaryFields = true;
//            cv::Mat inpainted = corrupted.clone();
//            maxwellHeavisidePDEInpainting(inpainted, mask, nIters, c_wave, dt, alpha, beta, gamma, useEquationOne, epsilon_0, mu_0, useEulerMethod, stationaryFields);
//            cv::imshow("maxwellHeavisidePDE", inpainted);
//            std::cout << "Results Maxwell-Heaviside equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, inpainted) << "/" <<
//                         cvlib::PSNR(input, inpainted) << "/" <<
//                         cvlib::SSIM(input, inpainted) << std::endl;
//        }

//        {
//            int N = 1000;  // Número de puntos espaciales
//            double dx = 0.01;  // Espaciado en x
//            double dy = 0.01;  // Espaciado en x
//            double nu = 0.001;  // Viscosidad
//            double dt = 0.001;  // Paso temporal
//            bool doUpwindDifferences = true;
//            cv::Mat inpainted = corrupted.clone();
//            burgersViscousEquationInpainting(inpainted, mask, nu, N, dt, dx, dy, doUpwindDifferences);
//            cv::imshow("burgersViscousEquationInpainting", inpainted);
//            std::cout << "Results Burguers viscous equation (MSE/PSNR/SSIM): " << std::fixed << std::setprecision(4) <<
//                         cvlib::MSE(input, inpainted) << "/" <<
//                         cvlib::PSNR(input, inpainted) << "/" <<
//                         cvlib::SSIM(input, inpainted) << std::endl;
//        }

//        {

//        }

//        cv::waitKey(0);
    }

    /// Edge detection
    std::cout << "\nEdge detection:" << std::endl;
//    {
//        cv::Mat input = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/peppers512.png", CV_IMWRITE_PAM_FORMAT_GRAYSCALE);
//        cv::resize(input, input, cv::Size(512, 512), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
//        input.convertTo(input, CV_64FC1, 1.0 / 255.0, 0.0);

////        cv::GaussianBlur(input, input, cv::Size(5, 5), 1, 1, cv::BORDER_REPLICATE);
//        input = cvlib::addGaussianNoise(input, 0.05);

//        cv::imshow("input", input);

//        {
//            cv::Mat edges;
//            cv::Mat input_8UC1;
//            input.convertTo(input_8UC1, CV_8UC1, 255.0, 0.0);
//            cv::Canny(input_8UC1, edges, 100, 200);
//            edges.convertTo(edges, CV_64FC1, 1.0 / 255.0, 0.0);
//            cv::imshow("Canny Edge Detection", edges);
//        }

//        {
//            cv::Mat input8U;
//            input.convertTo(input8U, CV_8U);
//            cv::Mat input64F;
//            input8U.convertTo(input64F, CV_64F, 1.0 / 255.0);
//            cv::Mat smoothed;
//            cv::GaussianBlur(input64F, smoothed, cv::Size(5, 5), 1.5);
//            cv::Mat laplacian;
//            cv::Laplacian(smoothed, laplacian, CV_64F, 3);
//            cv::Mat laplacianNormalized;
//            cv::normalize(laplacian, laplacianNormalized, 0, 1.0, cv::NORM_MINMAX);
//            cv::Mat laplacianNormalized8U;
//            laplacianNormalized.convertTo(laplacianNormalized8U, CV_8U, 255.0);
//            cv::Mat edges8U;
//            cv::threshold(laplacianNormalized8U, edges8U, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
//            cv::Mat edges64F;
//            edges8U.convertTo(edges64F, CV_64F, 1.0 / 255.0);
////            cv::imshow("Laplacian Normalized", laplacianNormalized8U);
//            cv::imshow("LoG Edge Detection", edges64F);
//        }

//        {
//            int iters = 5000;
//            double alpha = 0.01;
//            double dt = 0.001;
//            bool doUpwindDifferences = true;
//            cv::Mat filtered = input.clone();
//            kdvEquation2D(filtered, alpha, dt, iters, doUpwindDifferences);
//            cv::Mat grad_x, grad_y;
//            cv::Sobel(filtered, grad_x, CV_64F, 1, 0, 3);
//            cv::Sobel(filtered, grad_y, CV_64F, 0, 1, 3);
//            cv::Mat grad_magnitude;
//            cv::magnitude(grad_x, grad_y, grad_magnitude);
//            cv::Mat grad_magnitude_normalized;
//            grad_magnitude.convertTo(grad_magnitude_normalized, CV_64F, 1.0 / cv::norm(grad_magnitude, cv::NORM_INF));
//            cv::Mat grad_magnitude_normalized8U;
//            grad_magnitude_normalized.convertTo(grad_magnitude_normalized8U, CV_8U, 255.0);
//            cv::Mat edges8U;
//            cv::threshold(grad_magnitude_normalized8U, edges8U, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//            cv::Mat edges64F;
//            edges8U.convertTo(edges64F, CV_64F, 1.0 / 255.0);
//            cv::imshow("kdvEquation2D", filtered);
////            cv::imshow("kdvEquation2D Gradient Magnitude Normalized", grad_magnitude_normalized);
//            cv::imshow("kdvEquation2D Edge Detection", edges64F);
//        }

//        cv::waitKey(0);
//    }

    return 0;
}
