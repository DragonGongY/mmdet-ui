# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindowDL(object):
    def setupUi(self, MainWindowDL):
        MainWindowDL.setObjectName("MainWindowDL")
        MainWindowDL.resize(998, 727)
        self.centralwidget = QtWidgets.QWidget(MainWindowDL)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 511, 391))
        self.groupBox.setObjectName("groupBox")
        self.label_show = QtWidgets.QLabel(self.groupBox)
        self.label_show.setGeometry(QtCore.QRect(0, 20, 511, 371))
        self.label_show.setStyleSheet("background:rgb(0, 0, 0)")
        self.label_show.setScaledContents(True)
        self.label_show.setObjectName("label_show")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(540, 210, 211, 181))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_train = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_train.setGeometry(QtCore.QRect(20, 30, 80, 23))
        self.pushButton_train.setObjectName("pushButton_train")
        self.pushButton_test = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_test.setGeometry(QtCore.QRect(20, 70, 80, 23))
        self.pushButton_test.setObjectName("pushButton_test")
        self.pushButton_modify = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_modify.setGeometry(QtCore.QRect(120, 30, 80, 23))
        self.pushButton_modify.setObjectName("pushButton_modify")
        self.pushButton_finetune = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_finetune.setGeometry(QtCore.QRect(20, 110, 80, 23))
        self.pushButton_finetune.setObjectName("pushButton_finetune")
        self.pushButton_publish = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_publish.setGeometry(QtCore.QRect(120, 110, 81, 23))
        self.pushButton_publish.setObjectName("pushButton_publish")
        self.pushButton_stop = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_stop.setGeometry(QtCore.QRect(120, 70, 80, 23))
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.pushButton_readImage = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_readImage.setGeometry(QtCore.QRect(20, 150, 80, 23))
        self.pushButton_readImage.setObjectName("pushButton_readImage")
        self.pushButton_verify = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_verify.setGeometry(QtCore.QRect(120, 150, 80, 23))
        self.pushButton_verify.setObjectName("pushButton_verify")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 660, 511, 20))
        self.label.setObjectName("label")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(540, 0, 451, 91))
        self.groupBox_3.setObjectName("groupBox_3")
        self.comboBox_model = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_model.setGeometry(QtCore.QRect(10, 30, 431, 23))
        self.comboBox_model.setObjectName("comboBox_model")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_model.addItem("")
        self.comboBox_config = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_config.setGeometry(QtCore.QRect(10, 60, 431, 23))
        self.comboBox_config.setObjectName("comboBox_config")
        self.label_time = QtWidgets.QLabel(self.centralwidget)
        self.label_time.setGeometry(QtCore.QRect(540, 660, 411, 20))
        self.label_time.setObjectName("label_time")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 400, 511, 251))
        self.groupBox_4.setObjectName("groupBox_4")
        self.plainTextEdit_modifyConfig = QtWidgets.QPlainTextEdit(self.groupBox_4)
        self.plainTextEdit_modifyConfig.setGeometry(QtCore.QRect(0, 20, 511, 231))
        self.plainTextEdit_modifyConfig.setObjectName("plainTextEdit_modifyConfig")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(540, 400, 451, 251))
        self.groupBox_5.setObjectName("groupBox_5")
        self.plainTextEdit_log = QtWidgets.QPlainTextEdit(self.groupBox_5)
        self.plainTextEdit_log.setGeometry(QtCore.QRect(0, 20, 451, 231))
        self.plainTextEdit_log.setObjectName("plainTextEdit_log")
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(540, 90, 451, 61))
        self.groupBox_6.setObjectName("groupBox_6")
        self.comboBox_checkpoint = QtWidgets.QComboBox(self.groupBox_6)
        self.comboBox_checkpoint.setGeometry(QtCore.QRect(10, 30, 431, 23))
        self.comboBox_checkpoint.setObjectName("comboBox_checkpoint")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(760, 290, 231, 101))
        self.groupBox_7.setObjectName("groupBox_7")
        self.lineEdit_score = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit_score.setGeometry(QtCore.QRect(80, 30, 31, 23))
        self.lineEdit_score.setObjectName("lineEdit_score")
        self.label_score = QtWidgets.QLabel(self.groupBox_7)
        self.label_score.setGeometry(QtCore.QRect(10, 30, 71, 20))
        self.label_score.setObjectName("label_score")
        self.label_score_2 = QtWidgets.QLabel(self.groupBox_7)
        self.label_score_2.setGeometry(QtCore.QRect(10, 70, 101, 16))
        self.label_score_2.setObjectName("label_score_2")
        self.lineEdit_scorebbox = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit_scorebbox.setGeometry(QtCore.QRect(80, 70, 31, 23))
        self.lineEdit_scorebbox.setObjectName("lineEdit_scorebbox")
        self.label_trainval = QtWidgets.QLabel(self.groupBox_7)
        self.label_trainval.setGeometry(QtCore.QRect(120, 30, 61, 20))
        self.label_trainval.setObjectName("label_trainval")
        self.lineEdit_trainval = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit_trainval.setGeometry(QtCore.QRect(180, 30, 31, 23))
        self.lineEdit_trainval.setObjectName("lineEdit_trainval")
        self.label_train = QtWidgets.QLabel(self.groupBox_7)
        self.label_train.setGeometry(QtCore.QRect(140, 70, 41, 20))
        self.label_train.setObjectName("label_train")
        self.lineEdit_train = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit_train.setGeometry(QtCore.QRect(180, 70, 31, 23))
        self.lineEdit_train.setObjectName("lineEdit_train")
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_8.setGeometry(QtCore.QRect(760, 210, 231, 81))
        self.groupBox_8.setObjectName("groupBox_8")
        self.pushButton_convert2coco = QtWidgets.QPushButton(self.groupBox_8)
        self.pushButton_convert2coco.setGeometry(QtCore.QRect(30, 40, 80, 23))
        self.pushButton_convert2coco.setObjectName("pushButton_convert2coco")
        self.pushButton_convert2voc = QtWidgets.QPushButton(self.groupBox_8)
        self.pushButton_convert2voc.setGeometry(QtCore.QRect(130, 40, 80, 23))
        self.pushButton_convert2voc.setObjectName("pushButton_convert2voc")
        self.groupBox_9 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_9.setGeometry(QtCore.QRect(540, 150, 451, 61))
        self.groupBox_9.setObjectName("groupBox_9")
        self.comboBox_trainedmodel = QtWidgets.QComboBox(self.groupBox_9)
        self.comboBox_trainedmodel.setGeometry(QtCore.QRect(10, 30, 431, 23))
        self.comboBox_trainedmodel.setObjectName("comboBox_trainedmodel")
        MainWindowDL.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindowDL)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 998, 20))
        self.menubar.setObjectName("menubar")
        self.menuData = QtWidgets.QMenu(self.menubar)
        self.menuData.setObjectName("menuData")
        self.menuLabel = QtWidgets.QMenu(self.menubar)
        self.menuLabel.setObjectName("menuLabel")
        self.menuCheckPoints = QtWidgets.QMenu(self.menubar)
        self.menuCheckPoints.setObjectName("menuCheckPoints")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuWorkDir = QtWidgets.QMenu(self.menubar)
        self.menuWorkDir.setObjectName("menuWorkDir")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindowDL.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindowDL)
        self.statusbar.setObjectName("statusbar")
        MainWindowDL.setStatusBar(self.statusbar)
        self.actioncoco = QtWidgets.QAction(MainWindowDL)
        self.actioncoco.setObjectName("actioncoco")
        self.actionvoc = QtWidgets.QAction(MainWindowDL)
        self.actionvoc.setObjectName("actionvoc")
        self.actionCityScapes = QtWidgets.QAction(MainWindowDL)
        self.actionCityScapes.setObjectName("actionCityScapes")
        self.actionfaster_rcnn = QtWidgets.QAction(MainWindowDL)
        self.actionfaster_rcnn.setObjectName("actionfaster_rcnn")
        self.actionmask_rcnn = QtWidgets.QAction(MainWindowDL)
        self.actionmask_rcnn.setObjectName("actionmask_rcnn")
        self.actionssd = QtWidgets.QAction(MainWindowDL)
        self.actionssd.setObjectName("actionssd")
        self.actionyolo = QtWidgets.QAction(MainWindowDL)
        self.actionyolo.setObjectName("actionyolo")
        self.actionCityScapes_2 = QtWidgets.QAction(MainWindowDL)
        self.actionCityScapes_2.setObjectName("actionCityScapes_2")
        self.actionpascal_voc = QtWidgets.QAction(MainWindowDL)
        self.actionpascal_voc.setObjectName("actionpascal_voc")
        self.actionlabelme = QtWidgets.QAction(MainWindowDL)
        self.actionlabelme.setObjectName("actionlabelme")
        self.actionclear = QtWidgets.QAction(MainWindowDL)
        self.actionclear.setObjectName("actionclear")
        self.actioncocoNames = QtWidgets.QAction(MainWindowDL)
        self.actioncocoNames.setObjectName("actioncocoNames")
        self.actionvocNames = QtWidgets.QAction(MainWindowDL)
        self.actionvocNames.setObjectName("actionvocNames")
        self.actioncoco_detection = QtWidgets.QAction(MainWindowDL)
        self.actioncoco_detection.setObjectName("actioncoco_detection")
        self.actionvoc0712 = QtWidgets.QAction(MainWindowDL)
        self.actionvoc0712.setObjectName("actionvoc0712")
        self.actionshedules_1x = QtWidgets.QAction(MainWindowDL)
        self.actionshedules_1x.setObjectName("actionshedules_1x")
        self.actiondefault_runtime = QtWidgets.QAction(MainWindowDL)
        self.actiondefault_runtime.setObjectName("actiondefault_runtime")
        self.actionabout = QtWidgets.QAction(MainWindowDL)
        self.actionabout.setObjectName("actionabout")
        self.menuData.addAction(self.actioncoco)
        self.menuData.addAction(self.actionvoc)
        self.menuLabel.addAction(self.actionlabelme)
        self.menuCheckPoints.addAction(self.actioncoco_detection)
        self.menuCheckPoints.addAction(self.actionvoc0712)
        self.menuCheckPoints.addAction(self.actionshedules_1x)
        self.menuCheckPoints.addAction(self.actiondefault_runtime)
        self.menuWorkDir.addAction(self.actionclear)
        self.menuHelp.addAction(self.actionabout)
        self.menubar.addAction(self.menuData.menuAction())
        self.menubar.addAction(self.menuLabel.menuAction())
        self.menubar.addAction(self.menuCheckPoints.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuWorkDir.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindowDL)
        QtCore.QMetaObject.connectSlotsByName(MainWindowDL)

    def retranslateUi(self, MainWindowDL):
        _translate = QtCore.QCoreApplication.translate
        MainWindowDL.setWindowTitle(_translate("MainWindowDL", "Deep Learning General Platform"))
        self.groupBox.setTitle(_translate("MainWindowDL", "image show"))
        self.label_show.setText(_translate("MainWindowDL", "TextLabel"))
        self.groupBox_2.setTitle(_translate("MainWindowDL", "train and test"))
        self.pushButton_train.setText(_translate("MainWindowDL", "train"))
        self.pushButton_test.setText(_translate("MainWindowDL", "test"))
        self.pushButton_modify.setText(_translate("MainWindowDL", "modify"))
        self.pushButton_finetune.setText(_translate("MainWindowDL", "finetune"))
        self.pushButton_publish.setText(_translate("MainWindowDL", "publish"))
        self.pushButton_stop.setText(_translate("MainWindowDL", "stop"))
        self.pushButton_readImage.setText(_translate("MainWindowDL", "read"))
        self.pushButton_verify.setText(_translate("MainWindowDL", "verify"))
        self.label.setText(_translate("MainWindowDL", "www.bzlrobotics.com——Copyright 2018 - 2020 BZL. All Rights Reserved"))
        self.groupBox_3.setTitle(_translate("MainWindowDL", "config"))
        self.comboBox_model.setItemText(0, _translate("MainWindowDL", "cascade_rcnn"))
        self.comboBox_model.setItemText(1, _translate("MainWindowDL", "faster_rcnn"))
        self.comboBox_model.setItemText(2, _translate("MainWindowDL", "mask_rcnn"))
        self.comboBox_model.setItemText(3, _translate("MainWindowDL", "pascal_voc"))
        self.comboBox_model.setItemText(4, _translate("MainWindowDL", "ssd"))
        self.comboBox_model.setItemText(5, _translate("MainWindowDL", "yolo"))
        self.comboBox_model.setItemText(6, _translate("MainWindowDL", "pelee"))
        self.label_time.setText(_translate("MainWindowDL", "time display：2020:12:31"))
        self.groupBox_4.setTitle(_translate("MainWindowDL", "modify config"))
        self.groupBox_5.setTitle(_translate("MainWindowDL", "log"))
        self.groupBox_6.setTitle(_translate("MainWindowDL", "Pretrained_model"))
        self.groupBox_7.setTitle(_translate("MainWindowDL", "parameters"))
        self.lineEdit_score.setText(_translate("MainWindowDL", "0.5"))
        self.label_score.setText(_translate("MainWindowDL", "score:"))
        self.label_score_2.setText(_translate("MainWindowDL", "scorebbox:"))
        self.lineEdit_scorebbox.setText(_translate("MainWindowDL", "0.3"))
        self.label_trainval.setText(_translate("MainWindowDL", "trainval:"))
        self.lineEdit_trainval.setText(_translate("MainWindowDL", "0.9"))
        self.label_train.setText(_translate("MainWindowDL", "train:"))
        self.lineEdit_train.setText(_translate("MainWindowDL", "0.6"))
        self.groupBox_8.setTitle(_translate("MainWindowDL", "data convert"))
        self.pushButton_convert2coco.setText(_translate("MainWindowDL", "data2coco"))
        self.pushButton_convert2voc.setText(_translate("MainWindowDL", "data2voc"))
        self.groupBox_9.setTitle(_translate("MainWindowDL", "trained_model"))
        self.menuData.setTitle(_translate("MainWindowDL", "Data"))
        self.menuLabel.setTitle(_translate("MainWindowDL", "Label"))
        self.menuCheckPoints.setTitle(_translate("MainWindowDL", "Config"))
        self.menuTools.setTitle(_translate("MainWindowDL", "Tools"))
        self.menuWorkDir.setTitle(_translate("MainWindowDL", "WorkDir"))
        self.menuHelp.setTitle(_translate("MainWindowDL", "Help"))
        self.actioncoco.setText(_translate("MainWindowDL", "coco"))
        self.actionvoc.setText(_translate("MainWindowDL", "VOCdevkit"))
        self.actionCityScapes.setText(_translate("MainWindowDL", "CityScapes"))
        self.actionfaster_rcnn.setText(_translate("MainWindowDL", "faster_rcnn"))
        self.actionmask_rcnn.setText(_translate("MainWindowDL", "mask_rcnn"))
        self.actionssd.setText(_translate("MainWindowDL", "ssd"))
        self.actionyolo.setText(_translate("MainWindowDL", "yolo"))
        self.actionCityScapes_2.setText(_translate("MainWindowDL", "CityScapes"))
        self.actionpascal_voc.setText(_translate("MainWindowDL", "pascal_voc"))
        self.actionlabelme.setText(_translate("MainWindowDL", "labelme"))
        self.actionclear.setText(_translate("MainWindowDL", "clear"))
        self.actioncocoNames.setText(_translate("MainWindowDL", "cocoNames"))
        self.actionvocNames.setText(_translate("MainWindowDL", "vocNames"))
        self.actioncoco_detection.setText(_translate("MainWindowDL", "coco_detection"))
        self.actionvoc0712.setText(_translate("MainWindowDL", "voc0712"))
        self.actionshedules_1x.setText(_translate("MainWindowDL", "shedules_1x"))
        self.actiondefault_runtime.setText(_translate("MainWindowDL", "default_runtime"))
        self.actionabout.setText(_translate("MainWindowDL", "about"))

