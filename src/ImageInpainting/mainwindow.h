#pragma once

#include <QMainWindow>
#include <QtWidgets>

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
        QAction* mActionMaxwellHeavisideImageInpainting = nullptr;

        QAction* mActionAbout = nullptr;

        QAction* mRenderPencil = nullptr;
        QAction* mPencilProperties = nullptr;

        QAction* mActionToNoisyImage = nullptr;
        QAction* mActionToOriginalImage = nullptr;
        QAction* mActionToInpaintedImage = nullptr;
        QAction* mActionToMask = nullptr;
        QAction* mActionClearImage = nullptr;

        QToolBar* mToolbarFile = nullptr;
        QToolBar* mToolbarDrawInfo = nullptr;
        QToolBar* mToolbarImageStatus = nullptr;

        QLabel* mLabelOperationInfo = nullptr;

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

    private slots:
        void importImage();
        void exportImage();

        void transToNoisyImage();
        void transToOriginalImage();
        void transToInpaintedImage();
        void transToMask();
        void clearImage();

        void applyAlgorithm(QString algorithmName);

        void showNoiseWidget();
        void showMaxwellHeavisideInpaintingWidget();

        void setActionAndWidget(bool value1, bool value2);
        void needToResetImage();
        void about();
};
