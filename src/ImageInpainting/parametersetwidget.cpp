#include "parametersetwidget.h"

#include <QPushButton>
#include <QCheckBox>
#include <QLineEdit>
#include <QIntValidator>
#include <QComboBox>
#include <QGroupBox>

ParameterWidget::ParameterWidget(QWidget* parent, Parameter* parameter)
    : QWidget(parent)
{
    if(parameter == nullptr)
        return;

    mParameter = parameter;

    //    QVBoxLayout *VBoxLayout = new QVBoxLayout;
    //    this->setLayout(VBoxLayout);
    mLabel = new QLabel(mParameter->getLabel());
    this->setToolTip(mParameter->getToolTip());
    this->setWhatsThis(mParameter->getToolTip());
    //    VBoxLayout->addWidget(mLabel);
    switch (mParameter->getType())
    {
        case Parameter::kBool:
        {
            QCheckBox *CheckBox = new QCheckBox;
            //        VBoxLayout->addWidget(CheckBox);
            CheckBox->setChecked(mParameter->getValueBool());
            CheckBox->setText(mParameter->getLabel());
            mWidget = CheckBox;
            mWidgetType = kCheckBox;
            break;
        }
        case Parameter::kInt:
        {
            QLineEdit *LineEdit = new QLineEdit;
            if(mParameter->getHasValidator())
            {
                int minvalue = mParameter->getValidMin().toInt();
                int maxvalue = mParameter->getValidMax().toInt();
                LineEdit->setValidator(new QIntValidator(minvalue,maxvalue,this));
            }
            LineEdit->setText(QString::number(mParameter->getValueInt()));
            //        VBoxLayout->addWidget(LineEdit);
            mWidget = LineEdit;
            mWidgetType = kLineEdit;
            break;
        }
        case Parameter::kDouble:
        {
            QLineEdit* LineEdit = new QLineEdit;
            if (mParameter->getHasValidator())
            {
                double minvalue = mParameter->getValidMin().toDouble();
                double maxvalue = mParameter->getValidMax().toDouble();
                LineEdit->setValidator(new QDoubleValidator(minvalue,maxvalue,9,this));
            }
            LineEdit->setText(QString::number(mParameter->getValueDouble()));
            //        VBoxLayout->addWidget(LineEdit);
            mWidget = LineEdit;
            mWidgetType = kLineEdit;
            break;
        }
        case Parameter::kQStringList:
        {
            QComboBox* ComboBox = new QComboBox;
            ComboBox->addItems(mParameter->getValueQStringList());
            ComboBox->setCurrentIndex(mParameter->getStringListIndex());
            //        VBoxLayout->addWidget(ComboBox);
            mWidget = ComboBox;
            mWidgetType = kComboBox;
            break;
        }
        default:
            break;
    }
}

void ParameterWidget::setParameter()
{
    switch (mParameter->getType())
    {
        case Parameter::kInt:
        {
            QLineEdit* LineEdit = dynamic_cast<QLineEdit *>(mWidget);
            LineEdit->setText(QString::number(mParameter->getValueInt()));
            break;
        }
        case Parameter::kDouble:
        {
            QLineEdit* LineEdit = dynamic_cast<QLineEdit *>(mWidget);
            LineEdit->setText(QString::number(mParameter->getValueDouble()));
            break;
        }
        case Parameter::kBool:
        {
            QCheckBox* pCheckBox = dynamic_cast<QCheckBox *>(mWidget);
            pCheckBox->setChecked(mParameter->getValueBool());
            break;
        }
        default:
        {
            QComboBox* ComboBox = dynamic_cast<QComboBox *>(mWidget);
            ComboBox->setCurrentIndex(mParameter->getStringListIndex());
            break;
        }
    }

    update();
}

void ParameterWidget::resetParameter()
{
    switch (mParameter->getType())
    {
        case Parameter::kInt:
        {
            QLineEdit* LineEdit = dynamic_cast<QLineEdit *>(mWidget);
            LineEdit->setText(QString::number(mParameter->getDefaultValue().toInt()));
            break;
        }
        case Parameter::kDouble:
        {
            QLineEdit* LineEdit = dynamic_cast<QLineEdit *>(mWidget);
            LineEdit->setText(QString::number(mParameter->getDefaultValue().toDouble()));
            break;
        }
        case Parameter::kBool:
        {
            QCheckBox* pCheckBox = dynamic_cast<QCheckBox *>(mWidget);
            pCheckBox->setChecked(mParameter->getDefaultValue().toBool());
            break;
        }
        default:
        {
            QComboBox* ComboBox = dynamic_cast<QComboBox *>(mWidget);
            ComboBox->setCurrentIndex(mParameter->getDefaultIndex());
            break;
        }
    }

    update();
}

void ParameterWidget::updateParameter()
{
    switch (mParameter->getType())
    {
        case Parameter::kInt:
        {
            QLineEdit* LineEdit = dynamic_cast<QLineEdit *>(mWidget);
            QString t = LineEdit->text();
            bool ok;
            int value = t.toInt(&ok);
            if(ok)
                mParameter->setValue(value);
            break;
        }
        case Parameter::kDouble:
        {
            QLineEdit* LineEdit = dynamic_cast<QLineEdit *>(mWidget);
            QString t = LineEdit->text();
            bool ok;
            double value = t.toDouble(&ok);
            if(ok)
                mParameter->setValue(value);
            break;
        }
        case Parameter::kBool:
        {
            QCheckBox* CheckBox = dynamic_cast<QCheckBox *>(mWidget);
            mParameter->setValue(CheckBox->isChecked());
            break;
        }
        case Parameter::kQStringList:
        {
            QComboBox* ComboBox = dynamic_cast<QComboBox *>(mWidget);
            mParameter->setStringListIndex(ComboBox->currentIndex());
            break;
        }
        default:
            break;
    }
    update();
}

ParameterSetWidget::ParameterSetWidget(QWidget* parent, ParameterSet* parameterSet)
    : QDockWidget(parent)
{
    if (parameterSet == nullptr)
        return;

    mParameterSet = parameterSet;
    mParameterWidgetList.clear();

    mFrame = new QFrame;
    setWidget(mFrame);

    this->setStyleSheet("font-size:14px;");
    mVBoxLayout = new QVBoxLayout;
    mFrame->setLayout(mVBoxLayout);
    mLabel = new QLabel;
    mLabel->setWordWrap(true);
    mVBoxLayout->addWidget(mLabel);

    QPushButton* apply = new QPushButton(tr("Apply"), this);
    QObject::connect(apply, &QPushButton::clicked, this, &ParameterSetWidget::onApplyClick);
    mVBoxLayout->addWidget(apply);

    this->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    this->hide();
}

void ParameterSetWidget::onDefaultClick()
{
    for (int i = 0; i < (int)mParameterWidgetList.size(); i++)
        mParameterWidgetList[i]->resetParameter();
}

void ParameterSetWidget::onApplyClick()
{
    for (int i = 0; i < (int)mParameterWidgetList.size(); i++)
        mParameterWidgetList[i]->updateParameter();

    emit readyToApply(mParameterSet->getName());
}

void ParameterSetWidget::showWidget()
{
    this->setWindowTitle(mParameterSet->getLabel());
    mLabel->setText(mParameterSet->getIntroduction());

    QGroupBox* GroupBox = new QGroupBox("Setting");
    mVBoxLayoutSetting = new QVBoxLayout(GroupBox);
    QVector<Parameter *> all_parameters = mParameterSet->getAllParameters();
    for (int i = 0; i < (int)all_parameters.size(); i++)
    {
        Parameter *temp_parameter = all_parameters[i];
        ParameterWidget *mParameterwidget = new ParameterWidget(this, temp_parameter);
        mParameterWidgetList.append(mParameterwidget);
        //        mVBoxLayoutSetting->addWidget(mParameterwidget);
        if(mParameterwidget->mWidgetType != ParameterWidget::kCheckBox)
            mVBoxLayoutSetting->addWidget(mParameterwidget->getLabel());
        mVBoxLayoutSetting->addWidget(mParameterwidget->getWidget());
    }

    mVBoxLayoutSetting->setAlignment(Qt::AlignTop);
    mVBoxLayout->insertWidget(1, GroupBox);
    mVBoxLayout->setAlignment(Qt::AlignTop);
    mFrame->showNormal();
    mFrame->adjustSize();
    this->showNormal();
    this->adjustSize();
}
