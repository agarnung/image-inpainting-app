#include "iothread.h"

IOThread::IOThread(DataManager* dataManager)
    : mFileName{""}
    , mIOType{kNone}
{
    if (dataManager) mDataManager = dataManager;
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

    emit needToResetImage();
    emit setActionAndWidget(true, true);
}

void IOThread::exportImage(QString& fileName)
{
    emit statusShowMessage("Now writing fileName " + fileName + " ...");
    if(!mDataManager->exportImageToFile(fileName.toStdString()))
    {
        emit statusShowMessage("Writing fileName " + fileName + " failed.");
        return;
    }
    else
        emit statusShowMessage("Writing image " + fileName + " successful.");
}
