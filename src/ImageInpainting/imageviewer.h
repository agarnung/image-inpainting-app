#pragma once

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QColorDialog>
#include <QPen>

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

    protected:
        void mousePressEvent(QMouseEvent* event) override;
        void mouseMoveEvent(QMouseEvent* event) override;
        void mouseReleaseEvent(QMouseEvent* event) override;
        void wheelEvent(QWheelEvent* event) override;
        void mouseDoubleClickEvent(QMouseEvent* event) override;

    private:
        QGraphicsScene* mScene = nullptr;
        QGraphicsPixmapItem* mImageItem = nullptr;
        QPixmap mPixmap;
        QPainterPath mPath;
        QPen mPen;
        QColor mPencilColor;
        int mPencilSize;
        bool mDrawing;

        void init();

    public slots:
        void togglePencilDrawing();
        void setPencilColor();
};
