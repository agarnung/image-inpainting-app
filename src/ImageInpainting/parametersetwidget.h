#pragma once

#include "parameterset.h"

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QDockWidget>

/**
 * @class ParameterWidget
 * @brief Provides a GUI widget for editing algorithm parameters.
 *
 * Uses various input types such as checkboxes, combo boxes, and line edits.
 */
class ParameterWidget : public QWidget
{
    Q_OBJECT

    public:
        explicit ParameterWidget(QWidget* parent = nullptr, Parameter* parameter = nullptr);
        ~ParameterWidget() {};

        enum WidgetType{kNone, kCheckBox, kLineEdit, kComboBox};

        void setParameter();
        void resetParameter();
        void updateParameter();

        QLabel* getLabel() const { return mLabel;  }
        QWidget* getWidget() const { return mWidget; }

        WidgetType mWidgetType;

    private:
        Parameter* mParameter = nullptr;
        QLabel* mLabel = nullptr;
        QWidget* mWidget = nullptr;
};

class ParameterSetWidget : public QDockWidget
{
    Q_OBJECT

    public:
        ParameterSetWidget(QWidget* parent = nullptr, ParameterSet* parameterSet = nullptr);
        ~ParameterSetWidget() {};

        void showWidget();

    private:
        QFrame* mFrame = nullptr;
        ParameterSet* mParameterSet = nullptr;
        QVBoxLayout* mVBoxLayout = nullptr;
        QVBoxLayout* mVBoxLayoutSetting = nullptr;
        QLabel* mLabel = nullptr;
        QList<ParameterWidget *> mParameterWidgetList;

    public slots:
        void onDefaultClick();
        void onApplyClick();

    signals:
        void readyToApply(QString algoritmName);
        void transToNoisyImage();
        void transToOriginalImage();
        void transToInpaintedImage();
};
