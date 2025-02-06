#include "imageviewer.h"

ImageViewer::ImageViewer(QWidget* parent)
    : QGraphicsView{parent}
    , mPencilColor{Qt::red}
    , mPencilSize{5}
    , mDrawing{false}
{
    init();
}

void ImageViewer::updateImage(const QPixmap& image)
{
    mPixmap = image;
}

void ImageViewer::resetImage(const QPixmap& image)
{
    if (!image.isNull())
    {
        mImageItem = mScene->addPixmap(image);
        setSceneRect(image.rect());
    }
}

void ImageViewer::init()
{
    mScene = new QGraphicsScene(this);
    setScene(mScene);

    mPen.setColor(mPencilColor);
    mPen.setWidth(mPencilSize);
    mPen.setCapStyle(Qt::RoundCap);
    mPen.setJoinStyle(Qt::RoundJoin);
}

void ImageViewer::togglePencilDrawing()
{

}

void ImageViewer::setPencilColor()
{
    QColor color = QColorDialog::getColor(Qt::black, nullptr, tr("Set pencil color"));
    if (!color.isValid()) return;
    mPencilColor = color;
    this->update();
}

void ImageViewer::mousePressEvent(QMouseEvent* event)
{

}

void ImageViewer::mouseMoveEvent(QMouseEvent* event)
{

}

void ImageViewer::mouseReleaseEvent(QMouseEvent* event)
{

}

void ImageViewer::wheelEvent(QWheelEvent* event)
{
    // Zoom the view in the image, actually dont scale pencil, just the viewer sees the image more closely
}

void ImageViewer::mouseDoubleClickEvent(QMouseEvent* event)
{
    // Zoom in a predefined quantity or zoom out completely (toggle)
}
