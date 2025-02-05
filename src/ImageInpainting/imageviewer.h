#pragma once

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
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
        ImageViewer();
};
