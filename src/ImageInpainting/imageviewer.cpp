#include "imageviewer.h"
#include "qscrollbar.h"

ImageViewer::ImageViewer(QWidget* parent)
    : QGraphicsView{parent}
    , mPencilColor{Qt::red}
    , mPencilSize{5}
    , mDrawingActvated{false}
    , mIsUserDrawing{false}
    , mIsMiddleButtonPressed{false}
    , mIsAltPressed{false}
    , mIsShiftPressed{false}
{
    init();
}

void ImageViewer::init()
{
    mScene = new QGraphicsScene(this);
    setScene(mScene);

    setDragMode(QGraphicsView::NoDrag);
    setRenderHint(QPainter::SmoothPixmapTransform);

    setTransformationAnchor(QGraphicsView::AnchorViewCenter);

    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

    mPen.setColor(mPencilColor);
    mPen.setWidth(mPencilSize);
    mPen.setCapStyle(Qt::RoundCap);
    mPen.setJoinStyle(Qt::RoundJoin);
}

void ImageViewer::updateImage(const QPixmap& image)
{
    mPixmap = image;
    resetImage(image);
}

void ImageViewer::resetImage(const QPixmap& image)
{
    mScene->clear();
    mPixmap = image;

    if (!image.isNull())
    {
        mImageItem = mScene->addPixmap(image);
        setSceneRect(image.rect());
    }
}

void ImageViewer::togglePencilDrawing()
{
    mDrawingActvated = !mDrawingActvated;
}

void ImageViewer::setPencilColor()
{
    QColor color = QColorDialog::getColor(mPencilColor, this, tr("Select Pencil Color"));
    if (color.isValid())
    {
        mPencilColor = color;
        mPen.setColor(color);
    }
}

void ImageViewer::mousePressEvent(QMouseEvent* event)
{
    if (mDrawingActvated && event->button() == Qt::LeftButton)
    {
        mLastPoint = mapToScene(event->pos()).toPoint();
        mPath = QPainterPath();
        mPath.moveTo(mLastPoint);

        mPathItem = new QGraphicsPathItem();
        mPathItem->setPen(mPen);
        mScene->addItem(mPathItem);

        mIsUserDrawing = true;
    }

    if (event->button() == Qt::MiddleButton)
    {
        mIsMiddleButtonPressed = true;
        mPanStartPoint = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void ImageViewer::mouseMoveEvent(QMouseEvent* event)
{
    if (mIsMiddleButtonPressed)
    {
        QPoint delta = event->pos() - mPanStartPoint.toPoint();
        mPanStartPoint = event->pos();

        QScrollBar *hBar = horizontalScrollBar();
        QScrollBar *vBar = verticalScrollBar();
        hBar->setValue(hBar->value() - delta.x());
        vBar->setValue(vBar->value() - delta.y());
    }

    if (mDrawingActvated && mIsUserDrawing)
    {
        QPoint currentPoint = mapToScene(event->pos()).toPoint();
        mPath.lineTo(currentPoint);
        mPathItem->setPath(mPath);
    }
}

void ImageViewer::mouseReleaseEvent(QMouseEvent* event)
{
    if (mDrawingActvated && event->button() == Qt::LeftButton)
        mIsUserDrawing = false;

    if (event->button() == Qt::MiddleButton)
    {
        mIsMiddleButtonPressed = false;
        setCursor(Qt::ArrowCursor);
    }
}

void ImageViewer::clearDrawing()
{
    mScene->clear();
    if (!mPixmap.isNull())
        mImageItem = mScene->addPixmap(mPixmap);
}

void ImageViewer::wheelEvent(QWheelEvent* event)
{
    if (mIsAltPressed)
    {
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - event->angleDelta().x() / 12);
        return;
    }

    if (mIsShiftPressed)
    {
        verticalScrollBar()->setValue(verticalScrollBar()->value() - event->angleDelta().y() / 12);
        return;
    }

    QPointF scenePos = mapToScene(event->position().toPoint());
    const double scaleFactor = 1.15;
    if (event->angleDelta().y() > 0)
    {
        scale(scaleFactor, scaleFactor);
    }
    else
    {
        scale(1.0 / scaleFactor, 1.0 / scaleFactor);
    }

    QPointF newScenePos = mapToScene(event->position().toPoint());
    QPointF offset = newScenePos - scenePos;
    setSceneRect(sceneRect().translated(offset.x(), offset.y()));
}

void ImageViewer::mouseDoubleClickEvent(QMouseEvent* event)
{
    resetTransform();
    setSceneRect(mPixmap.rect());
    centerOn(mPixmap.rect().center());
}

void ImageViewer::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Alt)
        mIsAltPressed = true;
    else if (event->key() == Qt::Key_Shift)
        mIsShiftPressed = true;
}

void ImageViewer::keyReleaseEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Alt)
        mIsAltPressed = false;
    else if (event->key() == Qt::Key_Shift)
        mIsShiftPressed = false;
}

void ImageViewer::horizontalTranslation(int deltaX)
{
    horizontalScrollBar()->setValue(horizontalScrollBar()->value() - deltaX / 12);
}

void ImageViewer::verticalTranslation(int deltaY)
{
    verticalScrollBar()->setValue(verticalScrollBar()->value() - deltaY / 12);
}
