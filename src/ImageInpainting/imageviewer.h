#pragma once

#include "datamanager.h"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QColorDialog>
#include <QPen>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QInputDialog>

#include <QDialog>
#include <QColorDialog>
#include <QComboBox>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QVariant>
#include <QDialogButtonBox>

class PencilSettingsDialog : public QDialog
{
    Q_OBJECT

    public:
        PencilSettingsDialog(QWidget* parent = nullptr)
            : QDialog(parent)
            , mPencilColor(Qt::black)
            , mPencilSize(5)
            , mPenCapStyle(Qt::RoundCap)
            , mPenJoinStyle(Qt::RoundJoin)
        {
            QVBoxLayout* layout = new QVBoxLayout(this);

            QLabel* sizeLabel = new QLabel("Pencil Size:");
            layout->addWidget(sizeLabel);
            mSizeSpinBox = new QSpinBox();
            mSizeSpinBox->setRange(1, 100);
            mSizeSpinBox->setValue(mPencilSize);
            layout->addWidget(mSizeSpinBox);

            mColorButton = new QPushButton("Select Color");
            layout->addWidget(mColorButton);

            QLabel* capLabel = new QLabel("Cap Style:");
            layout->addWidget(capLabel);
            mCapComboBox = new QComboBox();
            mCapComboBox->addItem("Flat", Qt::FlatCap);
            mCapComboBox->addItem("Square", Qt::SquareCap);
            mCapComboBox->addItem("Round", Qt::RoundCap);
            layout->addWidget(mCapComboBox);

            QLabel* joinLabel = new QLabel("Join Style:");
            layout->addWidget(joinLabel);
            mJoinComboBox = new QComboBox();
            mJoinComboBox->addItem("Round", Qt::RoundJoin);
            mJoinComboBox->addItem("Miter", Qt::MiterJoin);
            mJoinComboBox->addItem("Bevel", Qt::BevelJoin);
            layout->addWidget(mJoinComboBox);

            QLabel* brushLabel = new QLabel("Brush Style:");
            layout->addWidget(brushLabel);
            mBrushComboBox = new QComboBox();
            mBrushComboBox->addItem("No Brush", QVariant::fromValue(Qt::NoBrush));
            mBrushComboBox->addItem("Solid Pattern", QVariant::fromValue(Qt::SolidPattern));
            mBrushComboBox->addItem("Dense1", QVariant::fromValue(Qt::Dense1Pattern));
            mBrushComboBox->addItem("Dense2", QVariant::fromValue(Qt::Dense2Pattern));
            mBrushComboBox->addItem("Dense3", QVariant::fromValue(Qt::Dense3Pattern));
            mBrushComboBox->addItem("Dense4", QVariant::fromValue(Qt::Dense4Pattern));
            mBrushComboBox->addItem("Dense5", QVariant::fromValue(Qt::Dense5Pattern));
            mBrushComboBox->addItem("Dense6", QVariant::fromValue(Qt::Dense6Pattern));
            mBrushComboBox->addItem("Dense7", QVariant::fromValue(Qt::Dense7Pattern));
            mBrushComboBox->addItem("Hor Pattern", QVariant::fromValue(Qt::HorPattern));
            mBrushComboBox->addItem("Ver Pattern", QVariant::fromValue(Qt::VerPattern));
            mBrushComboBox->addItem("Cross Pattern", QVariant::fromValue(Qt::CrossPattern));
            mBrushComboBox->addItem("BDiag Pattern", QVariant::fromValue(Qt::BDiagPattern));
            mBrushComboBox->addItem("FDiag Pattern", QVariant::fromValue(Qt::FDiagPattern));
            mBrushComboBox->addItem("DiagCross Pattern", QVariant::fromValue(Qt::DiagCrossPattern));
            mBrushComboBox->addItem("Linear Gradient", QVariant::fromValue(Qt::LinearGradientPattern));
            mBrushComboBox->addItem("Radial Gradient", QVariant::fromValue(Qt::RadialGradientPattern));
            mBrushComboBox->addItem("Conical Gradient", QVariant::fromValue(Qt::ConicalGradientPattern));
            mBrushComboBox->addItem("Texture", QVariant::fromValue(Qt::TexturePattern));
            layout->addWidget(mBrushComboBox);

            QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
            layout->addWidget(buttonBox);

            connect(buttonBox, &QDialogButtonBox::accepted, this, &PencilSettingsDialog::accept);
            connect(buttonBox, &QDialogButtonBox::rejected, this, &PencilSettingsDialog::reject);

            connect(mColorButton, &QPushButton::clicked, this, &PencilSettingsDialog::selectColor);
            connect(mSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &PencilSettingsDialog::updateSettings);
            connect(mCapComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PencilSettingsDialog::updateSettings);
            connect(mJoinComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PencilSettingsDialog::updateSettings);
            connect(mBrushComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &PencilSettingsDialog::updateSettings);
        }

        void accept() override
        {
            updateSettings();
            QDialog::accept();
        }

        void setPen(QPen& pen) const
        {
            pen.setWidth(mPencilSize);
            pen.setCapStyle(mPenCapStyle);
            pen.setJoinStyle(mPenJoinStyle);
            pen.setBrush(QBrush(static_cast<Qt::BrushStyle>(mBrushComboBox->currentData().toInt())));
            pen.setColor(mPencilColor); // Color goes after brush!
        }

        void setSettings(const QPen& pen)
        {
            mPencilColor = pen.color();
            mPencilSize = pen.width();
            mPenCapStyle = pen.capStyle();
            mPenJoinStyle = pen.joinStyle();

            mSizeSpinBox->setValue(mPencilSize);
            mCapComboBox->setCurrentIndex(mCapComboBox->findData(static_cast<int>(mPenCapStyle)));
            mJoinComboBox->setCurrentIndex(mJoinComboBox->findData(static_cast<int>(mPenJoinStyle)));

            mColorButton->setStyleSheet(QString("background-color: %1").arg(mPencilColor.name()));

            Qt::BrushStyle brushStyle = pen.brush().style();
            mBrushComboBox->setCurrentIndex(mBrushComboBox->findData(static_cast<int>(brushStyle)));

            this->update();
        }

    private slots:
        void selectColor()
        {
            QColorDialog::ColorDialogOptions options = QColorDialog::ShowAlphaChannel | QColorDialog::DontUseNativeDialog;
            QColor color = QColorDialog::getColor(mPencilColor, this, "Select Pencil Color", options);
            if (color.isValid())
            {
                mPencilColor = color;
                mColorButton->setStyleSheet(QString("background-color: %1").arg(mPencilColor.name()));
            }
        }

        void updateSettings()
        {
            mPencilSize = mSizeSpinBox->value();
            mPenCapStyle = static_cast<Qt::PenCapStyle>(mCapComboBox->currentData().toInt());
            mPenJoinStyle = static_cast<Qt::PenJoinStyle>(mJoinComboBox->currentData().toInt());
        }

    private:
        QColor mPencilColor;
        int mPencilSize;
        Qt::PenCapStyle mPenCapStyle;
        Qt::PenJoinStyle mPenJoinStyle;

        QSpinBox* mSizeSpinBox;
        QPushButton* mColorButton;
        QComboBox* mCapComboBox;
        QComboBox* mJoinComboBox;
        QComboBox* mBrushComboBox;
};

#include <QThread>

class MaskUpdater : public QThread
{
    Q_OBJECT

    public:
        MaskUpdater(QObject* parent = nullptr)
            : QThread(parent), mMask(nullptr) {}

        void setMask(QPixmap* mask, const QPoint& from, const QPoint& to, int penWidth)
        {
            mMask = mask;
            mFrom = from;
            mTo = to;
            mPenWidth = penWidth;
        }

    signals:
        void maskUpdated(const QPixmap& newMask);

    protected:
        void run() override
        {
            if (!mMask)
                return;

            QPixmap updatedMask = *mMask;
            QPainter painter(&updatedMask);
            painter.setPen(QPen(Qt::black, mPenWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
            painter.drawLine(mFrom, mTo);

            if (!updatedMask.isNull())
                emit maskUpdated(updatedMask);
        }

    private:
        QPixmap* mMask;
        QPoint mFrom, mTo;
        int mPenWidth;
};

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
    explicit ImageViewer(QWidget* parent = nullptr, DataManager* dataManager = nullptr);
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
        PencilSettingsDialog* mPencilSettingsDialog = nullptr;
        MaskUpdater* mMaskUpdater = nullptr;
        DataManager* mDataManager = nullptr;
        QGraphicsScene* mScene = nullptr;
        QGraphicsPixmapItem* mImageItem = nullptr;
        QGraphicsPathItem* mPathItem = nullptr;
        QPixmap mPixmap;
        QPixmap mInpaintingMask;
        QPen mPen;
        QPoint mLastPoint;
        QPointF mPanStartPoint;
        QPainterPath mPath;
        QList<QPainterPath> mDrawnPaths;
        bool mDrawingActivated;
        bool mIsUserDrawing;
        bool mIsMiddleButtonPressed;
        bool mIsAltPressed;
        bool mIsShiftPressed;

        void init();

        void horizontalTranslation(int deltaX);
        void verticalTranslation(int deltaY);

        void updateInpaintingMask(const QPoint& from, const QPoint& to);

    signals:
        void sendTimedMessage(const QString& msg, int duration_ms = 1000);

    public slots:
        void togglePencilDrawing();
        void showPencilSettingsDialog();
        void onMaskUpdated(const QPixmap& newMask);
};
