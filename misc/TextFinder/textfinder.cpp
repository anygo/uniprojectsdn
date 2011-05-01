#include "textfinder.h"
#include "ui_textfinder.h"

TextFinder::TextFinder(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TextFinder)
{
    ui->setupUi(this);
    ui->textEdit->setPlainText(QString("ich\ndu\ner\nsie\nes\nwir\nihr\nsie"));
}

TextFinder::~TextFinder()
{
    delete ui;
}


void TextFinder::on_lineEdit_textChanged(QString searchString)
{
    QPalette p;
    ui->textEdit->setPlainText(QString("ich\ndu\ner\nsie\nes\nwir\nihr\nsie"));

    if (ui->textEdit->find(searchString))
    {
        p.setColor(QPalette::Base, Qt::green);

    } else
    {
        p.setColor(QPalette::Base, Qt::red);
    }

    ui->lineEdit->setPalette(p);



}
