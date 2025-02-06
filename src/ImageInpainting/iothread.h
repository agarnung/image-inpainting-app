#pragma once

#include <QThread>
#include <QFileDialog>

#include "datamanager.h"

/**
 * @class IOThread
 * @brief Manages file I/O operations for images in a separate thread.
 *
 * Handles importing and exporting image files while keeping the GUI responsive.
 */
class IOThread : public QThread
{
    Q_OBJECT

    public:
        IOThread(DataManager* dataManager = nullptr);
        ~IOThread();

        enum ioType{kNone, kImport, kExport};

        QString mFileName;
        ioType mIOType;
        QWidget* mWidget = nullptr;
        DataManager* mDataManager = nullptr;

        void run();

        void importImage(QString& fileName);
        void exportImage(QString& fileName);

        inline void setFileName(const QString& fileName) { mFileName = fileName; }

    signals:
        void statusShowMessage(QString);
        void setActionAndWidget(bool, bool);
        void needToResetImage(bool);
};
