#ifndef TEXTFINDER_H
#define TEXTFINDER_H

#include <QWidget>

namespace Ui {
    class TextFinder;
}

class TextFinder : public QWidget
{
    Q_OBJECT

public:
    explicit TextFinder(QWidget *parent = 0);
    ~TextFinder();

private:
    Ui::TextFinder *ui;

private slots:
    void on_lineEdit_textChanged(QString );
};

#endif // TEXTFINDER_H
