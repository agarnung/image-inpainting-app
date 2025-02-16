#pragma once

#include <QMainWindow>
#include <QtWidgets>

#include <opencv4/opencv2/core.hpp>

class ImageViewer;
class DataManager;
class ParameterSet;
class ParameterSetWidget;
class CalculationThread;
class IOThread;

namespace Ui {
class MainWindow;
}

/**
 * @class MainWindow
 * @brief Represents the main window of the application.
 *
 * Responsible for containing and managing all the other GUI elements.
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

    public:
        explicit MainWindow(QWidget* parent = nullptr);
        ~MainWindow();

    private:
        Ui::MainWindow* mUi = nullptr;

        ImageViewer* mImageViewer = nullptr;
        DataManager* mDataManager = nullptr;
        ParameterSet* mParameterSet = nullptr;
        ParameterSetWidget* mParameterSetWidget = nullptr;
        CalculationThread* mCalculationThread = nullptr;
        IOThread* mIOThread = nullptr;

        QAction* mActionImportImage = nullptr;
        QAction* mActionExportImage = nullptr;
        QAction* mActionExit = nullptr;

        QAction* mActionNoise = nullptr;
        QAction* mActionTeleaImageInpainting = nullptr;
        QAction* mActionNavierStokesImageInpainting = nullptr;
        QAction* mActionMaxwellHeavisideImageInpainting = nullptr;
        QAction* mActionCahnHilliardImageInpainting = nullptr;
        QAction* mActionBurguersViscousImageInpainting = nullptr;
        QAction* mActionCriminisiImageInpainting = nullptr;
        QAction* mActionFastDigitalImageInpainting= nullptr;
        QAction* mActionLaplacianImageInpainting= nullptr;

        QAction* mActionAbout = nullptr;

        QAction* mActionRenderPencil = nullptr;
        QAction* mActionPencilProperties = nullptr;

        QAction* mActionToNoisyImage = nullptr;
        QAction* mActionToOriginalImage = nullptr;
        QAction* mActionToInpaintedImage = nullptr;
        QAction* mActionToMask = nullptr;
        QAction* mActionClearImage = nullptr;
        QAction* mActionResetDraw = nullptr;

        QToolBar* mToolbarFile = nullptr;
        QToolBar* mToolbarDrawInfo = nullptr;
        QToolBar* mToolbarImageStatus = nullptr;

        QLabel* mLabelOperationInfo = nullptr;
        QLabel* mLabelImageType = nullptr;
        QLabel* mLabelTimedMessages= nullptr;

        QMenu* mMenuFile= nullptr;
        QMenu* mMenuAlgorithms = nullptr;
        QMenu* mMenuHelp = nullptr;

        void init();
        void createActions();
        void createMenus();
        void createToolBars();
        void createStatusBar();

        void setActionStatus(bool value);

        void closeWidget();
        void showWidget();

        void setStyle();

    public slots:
        void receiveProcessImage(const cv::Mat& img);
        void receiveOtherMessage(const QString& msg);
        void receiveTimedMessage(const QString& msg, int duration_ms);

    private slots:
        void importImage();
        void exportImage();

        void transToNoisyImage();
        void transToOriginalImage();
        void transToInpaintedImage();
        void transToMask();
        void clearImage();
        void resetDraw();

        void applyAlgorithm(QString algorithmName);

        void showNoiseWidget();
        void showTeleaInpaintingWidget();
        void showNavierStokesInpaintingWidget();
        void showMaxwellHeavisideInpaintingWidget();
        void showCahnHilliardInpaintingWidget();
        void showBurguersViscousInpaintingWidget();
        void showCriminisiInpaintingWidget();
        void showFastDigitalInpaintingWidget();
        void showLaplacianInpaintingWidget();

        void setActionAndWidget(bool value1, bool value2);
        void needToResetImage();
        void about();
};
