#include "iothread.h"

IOThread::IOThread(DataManager* dataManager)
    : mFileName{""}
    , mIOType{kNone}
{
    if (dataManager) mDataManager = dataManager;
}

IOThread::~IOThread()
{
    ;
}

void IOThread::run()
{
    if(mIOType == kImport)
        importImage(mFileName);
    else
        exportImage(mFileName);
}

void IOThread::importImage(QString& fileName)
{
    emit setActionAndWidget(false, false);
    emit statusShowMessage("Now loading image " + fileName + " ...");
    if (!mDataManager->importImageFromFile(fileName.toStdString()))
    {
        emit statusShowMessage("Loading image " + fileName + " failed.");

        return;
    }
    else
        emit statusShowMessage("Loading image " + fileName + " successful.");

    emit needToResetImage(true);
    emit setActionAndWidget(true, true);
}

void IOThread::exportImage(QString& fileName)
{
    ;
}
