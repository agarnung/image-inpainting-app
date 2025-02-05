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
        QAction* mActionMaxwellHeavisideInpainting = nullptr;

        QAction* mActionAbout = nullptr;

        QAction* mRenderPencil = nullptr;

        QAction* mActionToNoisyImage = nullptr;
        QAction* mActionToOriginalImage = nullptr;
        QAction* mActionToInpaintedImage = nullptr;
        QAction* mActionClearImage = nullptr;

        QToolBar* mToolbarFile = nullptr;
        QToolBar* mToolbarDrawInfo = nullptr;
        QToolBar* mTtoolbarImageStatus = nullptr;

        QLabel* mLabelOperationInfo = nullptr;

        void init();
        void createActions();
        void createMenus();
        void createToolBars();
        void createStatusBar();

        void SetActionStatus(bool value);

    private slots:
        void ImportImage();
        void ExportImage();

        void setActionAndWidget(bool value1, bool value2);
        void needToUpdate(bool value);
        void About();
};
