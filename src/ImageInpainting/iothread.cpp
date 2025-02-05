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
    ;
}

void IOThread::exportImage(QString& fileName)
{
    ;
}
