#pragma once

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QColorDialog>
#include <QPen>
#include <QKeyEvent>
#include <QWheelEvent>

/**
 * @class ImageViewer
 * @brief Provides an Qt-based dynamic visualization of images.
 *
 * Allows users to interact with and modify the appearance of the image.
 */
class ImageViewer : public QGraphicsView
{
    Q_OBJECT

    public:
        explicit ImageViewer(QWidget* parent = nullptr);
        ~ImageViewer() {};

        void updateImage(const QPixmap& image);
        void resetImage(const QPixmap& image);
        void clearDrawing();

    protected:
        void mousePressEvent(QMouseEvent* event) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void mouseReleaseEvent(QMouseEvent* event) override;
        void wheelEvent(QWheelEvent* event) override;
        void mouseDoubleClickEvent(QMouseEvent* event) override;
        void keyPressEvent(QKeyEvent* event) override;
        void keyReleaseEvent(QKeyEvent* event) override;

    private:
        QGraphicsScene* mScene = nullptr;
        QGraphicsPixmapItem* mImageItem = nullptr;
        QGraphicsPathItem* mPathItem = nullptr;
        QPixmap mPixmap;
        QPoint mLastPoint;
        QPointF mPanStartPoint;
        QPainterPath mPath;
        QPen mPen;
        QColor mPencilColor;
        int mPencilSize;
        bool mDrawingActvated;
        bool mIsUserDrawing;
        bool mIsMiddleButtonPressed;
        bool mIsAltPressed;
        bool mIsShiftPressed;

        void init();

        void horizontalTranslation(int deltaX);
        void verticalTranslation(int deltaY);

    public slots:
        void togglePencilDrawing();
        void setPencilColor();
};
