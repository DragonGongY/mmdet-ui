from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, QDateTime, QThread
from PyQt5.QtGui import QPixmap, QIcon


import sys
import codecs
from PIL import Image
import threading
import shutil
import inspect
import ctypes
from datetime import datetime


from window.mainWindow import *
from window.train import *
from window.test import *
from window.utils import *
from window.oneImage import *
from window.publish import *
from window.script.labelme2coco import labelme2coco
from window.script.labelme2voc import labelme2voc

from pelee.train import train_pelee
from pelee.demo_img import verify_one_image

class MainWindow_Model(QMainWindow, Ui_MainWindowDL):

    train_end = pyqtSignal(str)
    finetune_train_end = pyqtSignal(str)
    def __init__(self):
        super(MainWindow_Model, self).__init__()
        self.setupUi(self)
        self.signal_connect()
        self.list_checkPoints()
        self.list_trainedModel()
        self.logger = get_logger("log_dir")
        self.time_update()
        self.__imgPath = None
        self.__cocodata_path = None
        self.__vocdata_path = None
        self.config_filePath = None
        self.train_thread = None

        self.coco_detection_path = "configs/_base_/datasets/coco_detection.py"
        self.voc0712_path = "configs/_base_/datasets/voc0712.py"
        self.schedule_1x_path = "configs/_base_/schedules/schedule_1x.py"
        self.default_runtime_path = "configs/_base_/default_runtime.py"

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(MainWindow_Model, cls).__new__(cls)
        return cls.instance

    def signal_connect(self):
        self.model_indexChanged()
        self.comboBox_model.currentIndexChanged.connect(self.model_indexChanged)
        self.comboBox_config.currentIndexChanged.connect(self.config_indexChanged)

        self.pushButton_train.clicked.connect(self.train_model)
        self.pushButton_test.clicked.connect(self.test_model)
        self.pushButton_modify.clicked.connect(self.modify_config)
        self.pushButton_verify.clicked.connect(self.verify_oneImage)
        self.pushButton_publish.clicked.connect(self.publish_model)
        self.pushButton_readImage.clicked.connect(self.read_image)
        self.pushButton_convert2coco.clicked.connect(self.data_convert2coco)
        self.pushButton_convert2voc.clicked.connect(self.data_convert2voc)
        self.pushButton_finetune.clicked.connect(self.finetune_Model)
        self.pushButton_stop.clicked.connect(self.stop_train)

        self.actioncoco.triggered.connect(self.cocodata_read)
        self.actionvoc.triggered.connect(self.vocdata_read)
        self.actionlabelme.triggered.connect(self.labelme_run)
        self.actionclear.triggered.connect(self.clear_workDir)
        self.actioncoco_detection.triggered.connect(lambda : self.list_configContent(self.coco_detection_path))
        self.actionvoc0712.triggered.connect(lambda: self.list_configContent(self.voc0712_path))
        self.actionshedules_1x.triggered.connect(lambda: self.list_configContent(self.schedule_1x_path))
        self.actiondefault_runtime.triggered.connect(lambda: self.list_configContent(self.default_runtime_path))
        self.actionabout.triggered.connect(self.show_about)

    def stop_train(self):
        self.stop_thread(self.train_thread)

    def _async_raise(self, tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def stop_thread(self, thread):
        self._async_raise(thread.ident, SystemExit)
        self.message_show("train thread suspend")

    def show_about(self):
        QMessageBox.information(self, "info", "version: 1.0\nos: Linux x64\nmmdetection: 2.0\n")

    def clear_workDir(self):
        workDir = "work_dirs/"
        shutil.rmtree(workDir)

    def list_trainedModel(self):
        self.comboBox_trainedmodel.clear()
        filePath = "model/"
        for maindir, subdir, file_name_list in os.walk(filePath):
            for filename in file_name_list:
                if ".pth" in filename:
                    self.comboBox_trainedmodel.addItem(filename.split('.')[0])

    def data_convert2voc(self):
        if self.__vocdata_path is None:
            QMessageBox.warning(self, "warning", "please input your image")
            return
        else:
            save_path = "data/VOCdevkit/"
            annotations_file = self.__vocdata_path + "/" + "class_names.txt"
            trainval_percent = float(self.lineEdit_trainval.text())
            train_percent = float(self.lineEdit_train.text())
            labelme2voc(self.__vocdata_path, save_path, annotations_file, trainval_percent, train_percent)
            self.write_class_names(annotations_file)
        self.message_show("data convert voc complished")

    def data_convert2coco(self):
        if self.__cocodata_path is None:
            QMessageBox.warning(self, "warning", "please input your image")
            return
        else:
            save_path = "data/coco/"
            annotations_file = self.__cocodata_path + "/" + "class_names.txt"
            trainval_percent = float(self.lineEdit_trainval.text())
            train_percent = float(self.lineEdit_train.text())
            labelme2coco(self.__cocodata_path, save_path, annotations_file, trainval_percent, train_percent)
            self.write_class_names(annotations_file)
        self.message_show("data convert coco complished")

    def write_class_names(self, class_names_file_path):
        class_names_count = 0
        for i, line in enumerate(open(class_names_file_path, 'r').readlines()):
            class_name = line.strip()
            if class_name == '__ignore__' or class_name == '_background_':
                continue
            else:
                class_names_count += 1
        file_path = "configs/_base_/models"
        for maindir, subdir, filename in os.walk(file_path):
            for file in filename:
                config = os.path.join(maindir, file)
                if "rpn" in config:
                    continue
                else:
                    cfg = mmcv.Config.fromfile(config)
                    if 'retinanet' in config or 'ssd' in config:
                        if isinstance(cfg.model.bbox_head, list):
                            for i in range(len(cfg.model.bbox_head)):
                                cfg.model.bbox_head[i].num_classes = class_names_count
                        else:
                            cfg.model.bbox_head.num_classes = class_names_count
                    else:
                        if isinstance(cfg.model.roi_head.bbox_head, list):
                            for i in range(len(cfg.model.roi_head.bbox_head)):
                                cfg.model.roi_head.bbox_head[i].num_classes = class_names_count
                        else:
                            cfg.model.roi_head.bbox_head.num_classes = class_names_count
                cfg.dump(config)

    def message_show(self, message):
        currentTime = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.plainTextEdit_log.appendPlainText("%s: %s" % (currentTime, message))
        self.logger.info('%s: %s' % (currentTime, message))

    def labelme_run(self):
        os.system("python ../labelme/__main__.py")

    def list_config_pelee(self):
        filePath = "../pelee/configs/"
        for maindir, subdir, file_name_list in os.walk(filePath):
            for filename in file_name_list:
                if filename == "Pelee_COCO.py" or filename == "Pelee_VOC.py":
                    self.comboBox_config.addItem(filename.split('.')[0])
                    configPath = os.path.join(maindir, filename)
        self.list_configContent(configPath)

    def list_config(self, name):
        filePath = "configs/" + name
        for maindir, subdir, file_name_list in os.walk(filePath):
            for filename in file_name_list:
                if ".py" in filename:
                    self.comboBox_config.addItem(filename.split('.')[0])
                    configPath = os.path.join(maindir, filename)
        self.list_configContent(configPath)

    def list_checkPoints(self):
        fileList = os.listdir("../checkpoints/")
        for file in fileList:
            self.comboBox_checkpoint.addItem(file.split('.')[0])

    def train_model(self):
        self.message_show("train model start")
        self.train_thread = threading.Thread(target=self.train_work, args=())
        self.train_thread.setDaemon(True)
        self.train_thread.start()

    def finetune_Model(self):
        self.message_show("finetune model start")
        t = threading.Thread(target=self.finetune_train_work, args=())
        t.setDaemon(True)
        t.start()

    def finetune_train_work(self):
        config = "configs/" + self.comboBox_model.currentText() + "/" + self.comboBox_config.currentText() + ".py"
        work_dir = "work_dirs"
        resume_from = "model/" + self.comboBox_checkpoint.currentText() + ".pth"
        finetune_train(config, work_dir, resume_from)
        self.finetune_train_end.connect(self.message_show)
        self.finetune_train_end.emit("finetune train end")

    def train_work(self):
        if self.comboBox_model.currentText() == "pelee":
            config = "../pelee/configs/" + self.comboBox_config.currentText() + ".py"
            train_pelee(config)
        else:
            config = "configs/" + self.comboBox_model.currentText() + "/" + self.comboBox_config.currentText() + ".py"
            work_dir = "work_dirs"
            train(config, work_dir)
        self.train_end.connect(self.message_show)
        self.train_end.emit("train model end")

    def test_model(self):
        config = "model/" + self.comboBox_config.currentText() + ".py"
        check_pointPath = "model/" + self.comboBox_config.currentText() + ".pth"
        result = "model/" + self.comboBox_config.currentText() + ".pkl"
        test(config, check_pointPath, result)
        self.message_show("pkl file saved")

    def list_configContent(self, file_path):
        self.config_filePath = file_path
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            contents = f.read()
            self.plainTextEdit_modifyConfig.setPlainText(contents)
            f.close()

    def modify_config(self):
        with codecs.open(self.config_filePath, 'w', encoding='utf-8') as f:
            contents = self.plainTextEdit_modifyConfig.toPlainText()
            f.write(contents)
            f.close()
            self.message_show("config content saved")

    def verify_oneImage(self):
        if self.__imgPath is None:
            QMessageBox.warning(self, "warning", "please input your image")
            return
        if self.comboBox_model.currentText() == "pelee":
            config = "model/" + self.comboBox_config.currentText() + '/' + self.comboBox_config.currentText() + ".py"
            trained_model = "model/" + self.comboBox_trainedmodel.currentText() + '/' + self.comboBox_trainedmodel.currentText() + ".pth"
            image = verify_one_image(self.__imgPath, config, trained_model)
        else:
            score_thr = float(self.lineEdit_score.text())
            score_bbox = float(self.lineEdit_scorebbox.text())
            checkpointPath = "model/" + self.comboBox_trainedmodel.currentText() + '/' + self.comboBox_trainedmodel.currentText() + ".pth"
            config = "model/" + self.comboBox_trainedmodel.currentText() + '/' + self.comboBox_trainedmodel.currentText() + ".py"
            img = str(self.__imgPath)
            model = init_detector(config, checkpointPath, device='cuda:0')
            result = inference_detector(model, img)
            image = show_result_pyplot(model, img, result, score_thr=score_thr)
        im = Image.fromarray(image)
        img_map = im.toqpixmap()
        img_map.scaled(self.label_show.size())
        self.label_show.setScaledContents(True)
        self.label_show.setPixmap(img_map)
        self.message_show("inference result is shown")

    def read_image(self):
        last_path = "image/"
        fileName = QFileDialog.getOpenFileName(self, "open_image", last_path, "Image Files(*.png *.jpg *.bmp)")
        self.__imgPath = fileName[0]
        self.message_show("read one image complished")

    def vocdata_read(self):
        last_path = "../data/"
        fileName = QFileDialog.getExistingDirectory(self, "choose your data path", last_path)
        self.__vocdata_path = fileName

    def cocodata_read(self):
        last_path = "../data/"
        fileName = QFileDialog.getExistingDirectory(self, "choose your data path", last_path)
        self.__cocodata_path = fileName

    def publish_model(self):
        if self.comboBox_model.currentText() == "pelee":
            model_path = "model/" + self.comboBox_config.currentText()
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            file_path = "../pelee/configs/" + self.comboBox_config.currentText() + ".py"
            final_file = model_path + "/" + self.comboBox_config.currentText() + ".py"
            shutil.copyfile(file_path, final_file, follow_symlinks=False)

            for maindir, subdir, file_name_list in os.walk("work_dirs/pelee/COCO"):
                pass
            out_file = model_path + "/" + self.comboBox_config.currentText() + ".pth"
            input_file = os.path.join(maindir, file_name_list[0])
            shutil.copyfile(input_file, out_file, follow_symlinks=False)

            class_file = "data/coco/class_names.txt"
            out_file = "model/" + self.comboBox_config.currentText() + "/" + "class_names.txt"
            shutil.copyfile(class_file, out_file, follow_symlinks=False)
        else:
            file_path = "work_dirs/" + self.comboBox_config.currentText()
            if os.path.exists(file_path):
                model_path = "model/" + self.comboBox_config.currentText()
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                in_file = file_path + '/' + 'latest.pth'
                out_file = model_path + "/" + self.comboBox_config.currentText() + ".pth"
                publish(in_file, out_file)
                config_file = file_path + '/' + self.comboBox_config.currentText() + ".py"
                final_file = model_path + "/" + self.comboBox_config.currentText() + ".py"
                shutil.copyfile(config_file, final_file, follow_symlinks=False)
        self.list_trainedModel()
        self.message_show(".pth and .py file published")

    def config_indexChanged(self):
        if self.comboBox_model.currentText() == "" or self.comboBox_config.currentText() == "":
            return
        elif "Pelee" in self.comboBox_config.currentText():
            file_path = "../pelee/configs/" + self.comboBox_config.currentText() + ".py"
        else:
            file_path = "configs/" + self.comboBox_model.currentText() + "/" + self.comboBox_config.currentText() + ".py"
        self.list_configContent(file_path)

    def model_indexChanged(self):
        self.comboBox_config.clear()
        name = self.comboBox_model.currentText()
        if name == "pelee":
            self.list_config_pelee()
        else:
            self.list_config(name)

    def time_update(self):
        self.back_thread = BackThread()
        self.back_thread.update_time.connect(self.time_display)
        self.back_thread.start()

    def time_display(self, currentTime):
        self.label_time.setText(currentTime)

class BackThread(QThread):
    update_time = pyqtSignal(str)

    def run(self):
        while (True):
            currentTime = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            self.update_time.emit(str(currentTime))
            time.sleep(1)  # sleep 1s

if __name__ == "__main__":
    app = QApplication(sys.argv)
    filename = "image/logo.ico"
    pixmap = QPixmap(filename)
    logo = QIcon(pixmap)
    app.setWindowIcon(logo)
    myWindow = MainWindow_Model()
    myWindow.__init__()
    myWindow.show()
    sys.exit(app.exec_())

