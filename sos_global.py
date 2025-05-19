#!/usr/bin/python
import os
import smtplib
import sys
import warnings
from datetime import UTC, datetime, timedelta
from email.mime.text import MIMEText

import pyproj
import ruamel.yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from PyQt6.QtCore import QMutex, QMutexLocker, QRect, QThread, QWaitCondition, pyqtSignal
from PyQt6.QtGui import QFont, QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QWidget,
)


def check_proj4_validity(string):
    try:
        pyproj.CRS.from_string(string)
        return True
    except pyproj.exceptions.CRSError:
        return False


class ComboCheckBox(QComboBox):
    ListChecked = pyqtSignal()

    def __init__(self, x, y, width, height, font, items, parent=None):
        super().__init__(parent)
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)
        self.items = ["全选"] + items
        self.box_list = []
        self.text = QLineEdit()
        self.state = 0
        q = QListWidget()
        for i in range(len(self.items)):
            self.box_list.append(QCheckBox())
            self.box_list[i].setText(self.items[i])
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.box_list[i])
            if i == 0:
                self.box_list[i].stateChanged.connect(self.all_selected)
            else:
                self.box_list[i].stateChanged.connect(self.show_selected)
            self.box_list[i].stateChanged.connect(self.not_empty_list)
        self.text.setReadOnly(True)
        self.setLineEdit(self.text)
        self.setModel(q.model())
        self.setView(q)

    def all_selected(self):
        if self.state == 0:
            self.state = 1
            for i in range(1, len(self.items)):
                self.box_list[i].setChecked(True)
        else:
            self.state = 0
            for i in range(1, len(self.items)):
                self.box_list[i].setChecked(False)
        self.show_selected()

    def get_selected(self):
        ret = []
        ret_idx = []
        for i in range(1, len(self.items)):
            if self.box_list[i].isChecked():
                ret.append(self.box_list[i].text())
                ret_idx.append(i)
        return ret, ret_idx

    def show_selected(self):
        self.text.clear()
        ret, ret_idx = self.get_selected()
        ret_text = "; ".join(ret)
        self.text.setText(ret_text)

    def not_empty_list(self):
        self.ListChecked.emit()

    def new_list(self, new_items):
        self.clear()
        self.button.setEnabled(False)
        self.items = ["全选"] + new_items
        self.box_list = []
        self.text = QLineEdit()
        self.state = 0
        q = QListWidget()
        for i in range(len(self.items)):
            self.box_list.append(QCheckBox())
            self.box_list[i].setText(self.items[i])
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.box_list[i])
            if i == 0:
                self.box_list[i].stateChanged.connect(self.all_selected)
            else:
                self.box_list[i].stateChanged.connect(self.show_selected)
            self.box_list[i].stateChanged.connect(self.not_empty_list)
        self.text.setReadOnly(True)
        self.setLineEdit(self.text)
        self.setModel(q.model())
        self.setView(q)


class ComboBox(QComboBox):
    def __init__(self, x, y, width, height, font, need_none, items, parent=None):
        super().__init__(parent)
        self.need_none = need_none
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)
        if need_none:
            self.addItem("无")
        for item in items:
            self.addItem(item)


class Label(QLabel):
    def __init__(self, x, y, width, height, font, text, parent=None):
        super().__init__(parent)
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)
        self.setText(text)


class LineEdit(QLineEdit):
    def __init__(self, x, y, width, height, font, parent=None):
        super().__init__(parent)
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)


class RadioButton(QRadioButton):
    def __init__(self, x, y, width, height, font, text, parent=None):
        super().__init__(parent)
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)
        self.setText(text)


class Button(QPushButton):
    def __init__(self, x, y, width, height, font, text, parent=None):
        super().__init__(parent)
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)
        self.setText(text)


class UiGeo(QWidget):
    def __init__(self):
        super().__init__()
        with open("sos_global_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            self.config_data = yaml.load(yaml_file)

        satpy_config_path = self.config_data["basic_path"]["satpy_config_path"]

        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        os.environ["DASK_ARRAY__CHUNK_SIZE"] = "16MiB"
        os.environ["DASK_NUM_WORKERS"] = "12"
        os.environ["PSP_CONFIG_FILE"] = os.path.join(satpy_config_path, "pyspectral/pyspectral.yaml")

        self.sat_dict = self.config_data["satellites"]
        init_proj_dict = self.config_data["projection"][list(self.sat_dict)[0]]
        init_comp_dict = self.config_data["composites"][list(self.sat_dict)[0]]

        font = QFont()
        font.setPointSize(15)
        font.setBold(True)
        font1 = QFont()
        font1.setPointSize(14)
        font2 = QFont()
        font2.setPointSize(13)
        font2.setBold(True)
        font3 = QFont()
        font3.setPointSize(10)
        font3.setBold(True)

        self.setWindowTitle("S.O.S Toolbox")
        self.resize(1200, 1000)

        self.tab = QTabWidget(self)
        self.tab.setGeometry(QRect(0, 0, 1200, 1000))
        self.process_tab = QWidget()
        self.tab.addTab(self.process_tab, "处理")

        self.start_button_geo = Button(830, 210, 160, 100, font, "开始", self.process_tab)
        self.start_button_geo.setEnabled(False)
        self.start_button_geo.clicked.connect(self.geo_start)

        self.radio_step_geo = RadioButton(730, 150, 200, 30, font1, "按10分钟节点步进", self.process_tab)
        self.radio_batch_geo = RadioButton(930, 150, 200, 30, font1, "批量处理", self.process_tab)
        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.radio_step_geo)
        self.button_group.addButton(self.radio_batch_geo)
        self.radio_step_geo.setChecked(True)

        self.label_title_geo = Label(340, 20, 740, 65, font, "在同目录下的sos_global_config.yaml中编辑参数并重启程序",
                                     self.process_tab)

        self.label_start_date_dl = Label(60, 360 + 270, 250, 40, font2, "观测起始日期8位数字", self.process_tab)
        self.label_start_time_dl = Label(330, 360 + 270, 190, 40, font2, "观测起始时间4位数字", self.process_tab)
        self.label_end_date_dl = Label(60, 460 + 270, 250, 40, font2, "观测结束日期8位数字", self.process_tab)
        self.label_end_time_dl = Label(330, 460 + 270, 190, 40, font2, "观测结束时间4位数字", self.process_tab)

        self.lineEdit_start_date_dl = LineEdit(60, 400 + 270, 250, 40, font1, self.process_tab)
        self.lineEdit_start_time_dl = LineEdit(330, 400 + 270, 190, 40, font1, self.process_tab)
        self.lineEdit_end_date_dl = LineEdit(60, 500 + 270, 250, 40, font1, self.process_tab)
        self.lineEdit_end_time_dl = LineEdit(330, 500 + 270, 190, 40, font1, self.process_tab)

        self.lineEdit_start_date_dl.setMaxLength(8)
        self.lineEdit_start_time_dl.setMaxLength(4)
        self.lineEdit_end_date_dl.setMaxLength(8)
        self.lineEdit_end_time_dl.setMaxLength(4)

        int_validator = QIntValidator()
        self.lineEdit_start_date_dl.setValidator(int_validator)
        self.lineEdit_start_time_dl.setValidator(int_validator)
        self.lineEdit_end_date_dl.setValidator(int_validator)
        self.lineEdit_end_time_dl.setValidator(int_validator)

        self.lineEdit_start_date_dl.setText(datetime(datetime.now().year, 1, 1).strftime("%Y%m%d"))
        self.lineEdit_start_time_dl.setText("0000")
        self.lineEdit_end_date_dl.setText(datetime(datetime.now().year, 1, 1).strftime("%Y%m%d"))
        self.lineEdit_end_time_dl.setText("0010")

        init_proj_list = self.common_items_with_same_index("projection")
        init_proj_list = [(init_proj_dict[key]["name"] + " | " + init_proj_dict[key]["area"] + " | " + "中央经线" +
                           str(init_proj_dict[key]["center_lon"]) + " | " + "分辨率" + str(init_proj_dict[key]["res"]))
                          for key in init_proj_list]

        self.cb_proj_geo = ComboBox(60, 240, 600, 40, font2, False, init_proj_list, self.process_tab)
        self.cb_proj_geo.currentIndexChanged.connect(self.area_select)

        init_comp_list = self.common_items_with_same_index("composites")
        init_comp_list = [init_comp_dict[key]["name"] for key in init_comp_list]
        self.cb_product_geo = ComboCheckBox(60, 330, 600, 40, font2, init_comp_list, parent=self.process_tab)
        self.cb_product_geo.ListChecked.connect(self.valid_dt_and_items_geo)

        self.lineEdit_start_date_dl.textChanged.connect(self.valid_dt_and_items_geo)
        self.lineEdit_start_time_dl.textChanged.connect(self.valid_dt_and_items_geo)
        self.lineEdit_end_date_dl.textChanged.connect(self.valid_dt_and_items_geo)
        self.lineEdit_end_time_dl.textChanged.connect(self.valid_dt_and_items_geo)

        self.label_proj_geo = Label(60, 200, 200, 40, font1, "选择投影", self.process_tab)
        self.label_product_geo = Label(60, 290, 200, 40, font1, "选择产品", self.process_tab)

        self.lineEdit_w_geo = LineEdit(60, 450, 190, 40, font1, self.process_tab)
        self.lineEdit_h_geo = LineEdit(260, 450, 190, 40, font1, self.process_tab)
        self.lineEdit_xy_geo = LineEdit(480, 450, 520, 40, font1, self.process_tab)
        self.lineEdit_area_geo = LineEdit(1020, 450, 110, 40, font1, self.process_tab)
        self.lineEdit_proj4_geo = LineEdit(60, 540, 580, 40, font1, self.process_tab)
        self.lineEdit_proj_date_geo = LineEdit(660, 540, 470, 40, font1, self.process_tab)

        self.lineEdit_w_geo.setReadOnly(True)
        self.lineEdit_h_geo.setReadOnly(True)
        self.lineEdit_xy_geo.setReadOnly(True)
        self.lineEdit_area_geo.setReadOnly(True)
        self.lineEdit_proj4_geo.setReadOnly(True)
        self.lineEdit_proj_date_geo.setReadOnly(True)

        init_area_id = init_proj_dict[list(init_proj_dict)[0]]["area_id"]
        init_width = init_proj_dict[list(init_proj_dict)[0]]["width"]
        init_height = init_proj_dict[list(init_proj_dict)[0]]["height"]
        init_proj = init_proj_dict[list(init_proj_dict)[0]]["proj"]
        init_extent = init_proj_dict[list(init_proj_dict)[0]]["extent"]
        init_date = datetime.strptime(str(init_proj_dict[list(init_proj_dict)[0]]["date"]),
                                      "%Y%m%d%H%M%S")
        init_date_str = datetime.strftime(init_date, "UTC时间 %Y年%m月%d日%H时%M分")
        self.lineEdit_w_geo.setText(str(init_width))
        self.lineEdit_h_geo.setText(str(init_height))
        self.lineEdit_xy_geo.setText(str(init_extent))
        self.lineEdit_area_geo.setText(str(init_area_id))
        self.lineEdit_proj4_geo.setText(str(init_proj))
        self.lineEdit_proj_date_geo.setText(init_date_str)

        self.label_wh_geo = Label(60, 410, 390, 40, font2, "宽 高", self.process_tab)
        self.label_xy_geo = Label(480, 410, 520, 40, font2, "非经纬度xy坐标 (min_x  min_y  max_x  max_y)",
                                  self.process_tab)
        self.label_area_geo = Label(1020, 410, 110, 40, font2, "区域代码", self.process_tab)
        self.label_proj4_geo = Label(60, 500, 120, 40, font2, "Proj4表达式", self.process_tab)
        self.label_proj_date_geo = Label(660, 500, 120, 40, font2, "添加时间", self.process_tab)

        self.mutex = QMutex()
        self.condition = QWaitCondition()

        self.geo_thread = DownloadAndGeoThread(True, 0, 0, [1], "202309200000", "202309200010",
                                               self.mutex, self.condition)
        self.clean_thread = CleanThread(0, [1], self.mutex, self.condition)

        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.auto_update, trigger=CronTrigger(minute="0/10"))

        self.auto_update_start_button_geo = Button(730, 330, 160, 30, font, "开始自动更新", self.process_tab)
        self.auto_update_start_button_geo.setEnabled(False)
        self.auto_update_start_button_geo.clicked.connect(self.auto_update_start)

        self.auto_update_end_button_geo = Button(930, 330, 160, 30, font, "停止自动更新", self.process_tab)
        self.auto_update_end_button_geo.setEnabled(False)
        self.auto_update_end_button_geo.clicked.connect(self.auto_update_end)

    def valid_dt_and_items_geo(self):
        ret, ret_idx = self.cb_product_geo.get_selected()
        valid_items = True if ret_idx else False

        if (len(self.lineEdit_start_date_dl.text()) == 8 and len(self.lineEdit_start_time_dl.text()) == 4 and
                len(self.lineEdit_end_date_dl.text()) == 8 and len(self.lineEdit_end_time_dl.text())) == 4:
            valid_length = True
        else:
            valid_length = False

        start_dt = self.lineEdit_start_date_dl.text() + self.lineEdit_start_time_dl.text()
        end_dt = self.lineEdit_end_date_dl.text() + self.lineEdit_end_time_dl.text()

        try:
            start_date = datetime.strptime(start_dt, "%Y%m%d%H%M")
            end_date = datetime.strptime(end_dt, "%Y%m%d%H%M")
            valid_date = True
            valid_range = True if end_date - start_date >= timedelta(minutes=10) else False

        except ValueError:
            valid_date = valid_range = False

        self.start_button_geo.setEnabled(all([valid_items, valid_length, valid_date, valid_range]))
        self.auto_update_start_button_geo.setEnabled(all([valid_items, valid_length, valid_date, valid_range]))

    def auto_update_start(self):
        self.start_button_geo.setEnabled(False)
        self.auto_update_start_button_geo.setEnabled(False)

        self.cb_proj_geo.setEnabled(False)
        self.cb_product_geo.setEnabled(False)

        self.lineEdit_start_date_dl.setEnabled(False)
        self.lineEdit_start_time_dl.setEnabled(False)
        self.lineEdit_end_date_dl.setEnabled(False)
        self.lineEdit_end_time_dl.setEnabled(False)

        self.radio_step_geo.setChecked(True)
        self.radio_step_geo.setEnabled(False)
        self.radio_batch_geo.setEnabled(False)

        self.scheduler.start()

        if not self.geo_thread.isRunning() and not self.clean_thread.isRunning():
            self.auto_update_end_button_geo.setEnabled(True)

    def auto_update_end(self):
        self.auto_update_end_button_geo.setEnabled(False)
        self.scheduler.shutdown()

        if not self.geo_thread.isRunning() and not self.clean_thread.isRunning():
            self.auto_update_start_button_geo.setEnabled(True)
            self.start_button_geo.setEnabled(True)

            self.cb_proj_geo.setEnabled(True)
            self.cb_product_geo.setEnabled(True)

            self.lineEdit_start_date_dl.setEnabled(True)
            self.lineEdit_start_time_dl.setEnabled(True)
            self.lineEdit_end_date_dl.setEnabled(True)
            self.lineEdit_end_time_dl.setEnabled(True)

            self.radio_step_geo.setEnabled(True)
            self.radio_batch_geo.setEnabled(True)

    def auto_update(self):
        current_time = datetime.now(UTC) - timedelta(minutes=30)
        current_time_10min = current_time + timedelta(minutes=10)
        self.lineEdit_start_date_dl.setText(current_time.strftime("%Y%m%d"))
        self.lineEdit_start_time_dl.setText(current_time.strftime("%H%M"))
        self.lineEdit_end_date_dl.setText(current_time_10min.strftime("%Y%m%d"))
        self.lineEdit_end_time_dl.setText(current_time_10min.strftime("%H%M"))

        sat_index = 0
        proj_index = self.cb_proj_geo.currentIndex()
        product_index_list = self.cb_product_geo.get_selected()[1] if (
                len(self.cb_product_geo.get_selected()[1]) > 0) else [1]
        start_date = self.lineEdit_start_date_dl.text() + self.lineEdit_start_time_dl.text()
        end_date = self.lineEdit_end_date_dl.text() + self.lineEdit_end_time_dl.text()

        self.geo_thread = DownloadAndGeoThread(True, sat_index, proj_index, product_index_list, start_date, end_date,
                                               self.mutex, self.condition)

        self.geo_thread.start_date_corr.connect(self.start_date_corr)
        self.geo_thread.start_time_corr.connect(self.start_time_corr)
        self.geo_thread.end_date_corr.connect(self.end_date_corr)
        self.geo_thread.end_time_corr.connect(self.end_time_corr)
        self.geo_thread.clean_dict.connect(self.geo_done_event)

        self.geo_thread.start()

        self.start_button_geo.setEnabled(False)
        self.auto_update_end_button_geo.setEnabled(True)

    def common_items_with_same_index(self, dict_key):
        item_sets = []
        for sat in self.config_data["satellites"]:
            satellite = {"GOES-WEST": "GOES-WEST",
                         "GOES-EAST": "GOES-WEST",
                         "MTG-FD": "MTG-FD",
                         "MTG-EU": "MTG-FD"}.get(sat, sat)
            for i in list(self.config_data[dict_key][satellite]):
                item_sets.append(i)

        common_items = []
        for j in item_sets:
            if j not in common_items:
                common_items.append(j)

        common_items_with_same_index = []
        for common_item in common_items:
            indices = set()
            for sat in self.config_data["satellites"]:
                satellite = {"GOES-WEST": "GOES-WEST",
                             "GOES-EAST": "GOES-WEST",
                             "MTG-FD": "MTG-FD",
                             "MTG-EU": "MTG-FD"}.get(sat, sat)
                idx = list(self.config_data[dict_key][satellite]).index(common_item)
                indices.add(idx)

            if len(indices) == 1:
                common_items_with_same_index.append(common_item)

        return common_items_with_same_index

    def area_select(self):
        idx = 0
        proj_dict = self.config_data["projection"][list(self.sat_dict)[idx]]
        idx2 = self.cb_proj_geo.currentIndex()
        area_id = proj_dict[list(proj_dict)[idx2]]["area_id"]
        width = proj_dict[list(proj_dict)[idx2]]["width"]
        height = proj_dict[list(proj_dict)[idx2]]["height"]
        proj = proj_dict[list(proj_dict)[idx2]]["proj"]
        extent = proj_dict[list(proj_dict)[idx2]]["extent"]
        date = datetime.strptime(str(proj_dict[list(proj_dict)[idx2]]["date"]), "%Y%m%d%H%M%S")
        date_str = datetime.strftime(date, "UTC时间 %Y年%m月%d日%H时%M分")
        self.lineEdit_w_geo.setText(str(width))
        self.lineEdit_h_geo.setText(str(height))
        self.lineEdit_xy_geo.setText(str(extent))
        self.lineEdit_area_geo.setText(str(area_id))
        self.lineEdit_proj4_geo.setText(str(proj))
        self.lineEdit_proj_date_geo.setText(date_str)

    def geo_start(self):
        self.start_button_geo.setEnabled(False)
        self.auto_update_start_button_geo.setEnabled(False)

        self.cb_proj_geo.setEnabled(False)
        self.cb_product_geo.setEnabled(False)

        self.lineEdit_start_date_dl.setEnabled(False)
        self.lineEdit_start_time_dl.setEnabled(False)
        self.lineEdit_end_date_dl.setEnabled(False)
        self.lineEdit_end_time_dl.setEnabled(False)

        self.radio_step_geo.setEnabled(False)
        self.radio_batch_geo.setEnabled(False)

        sat_index = 0
        proj_index = self.cb_proj_geo.currentIndex()
        product_index_list = self.cb_product_geo.get_selected()[1]
        start_date = self.lineEdit_start_date_dl.text() + self.lineEdit_start_time_dl.text()
        end_date = self.lineEdit_end_date_dl.text() + self.lineEdit_end_time_dl.text()

        step_mode = True if self.radio_step_geo.isChecked() else False
        self.geo_thread = DownloadAndGeoThread(step_mode, sat_index, proj_index, product_index_list,
                                               start_date, end_date, self.mutex, self.condition)

        self.geo_thread.start_date_corr.connect(self.start_date_corr)
        self.geo_thread.start_time_corr.connect(self.start_time_corr)
        self.geo_thread.end_date_corr.connect(self.end_date_corr)
        self.geo_thread.end_time_corr.connect(self.end_time_corr)
        self.geo_thread.clean_dict.connect(self.geo_done_event)

        self.geo_thread.start()

    def geo_done_event(self, clean_dict):
        self.geo_thread.quit()

        sat_index = 0
        product_index_list = self.cb_product_geo.get_selected()[1]

        self.clean_thread = CleanThread(sat_index, product_index_list, self.mutex, self.condition, clean_dict)
        self.clean_thread.done.connect(self.clean_done_event)
        self.clean_thread.start()

    def clean_done_event(self):
        self.clean_thread.quit()

        if self.scheduler.state == 0:
            self.auto_update_start_button_geo.setEnabled(True)
            self.start_button_geo.setEnabled(True)

            self.cb_proj_geo.setEnabled(True)
            self.cb_product_geo.setEnabled(True)

            self.lineEdit_start_date_dl.setEnabled(True)
            self.lineEdit_start_time_dl.setEnabled(True)
            self.lineEdit_end_date_dl.setEnabled(True)
            self.lineEdit_end_time_dl.setEnabled(True)

            self.radio_step_geo.setEnabled(True)
            self.radio_batch_geo.setEnabled(True)

    def start_date_corr(self, string):
        self.lineEdit_start_date_dl.setText(string)

    def start_time_corr(self, string):
        self.lineEdit_start_time_dl.setText(string)

    def end_date_corr(self, string):
        self.lineEdit_end_date_dl.setText(string)

    def end_time_corr(self, string):
        self.lineEdit_end_time_dl.setText(string)


class DownloadAndGeoThread(QThread):
    start_date_corr = pyqtSignal(str)
    start_time_corr = pyqtSignal(str)
    end_date_corr = pyqtSignal(str)
    end_time_corr = pyqtSignal(str)
    clean_dict = pyqtSignal(dict)
    last_round = pyqtSignal(bool)

    def __init__(self, step_mode: bool, sat_index, proj_index, product_index_list, start_date, end_date,
                 mutex: QMutex, condition: QWaitCondition):
        super(DownloadAndGeoThread, self).__init__()
        self.step_mode = step_mode
        self.sat_index = sat_index
        self.proj_index = proj_index
        self.product_index_list = [max(0, idx - 1) for idx in product_index_list]
        self.start_date = start_date
        self.end_date = end_date
        self.mutex = mutex
        self.condition = condition

    def run(self):
        def send_mail(topic: str, message: str):
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "lpeacemaker711@gmail.com"
            app_password = "ktohlntvqwxjpkbk"
            receiver_email = "lpeacemaker711@gmail.com"

            msg = MIMEText(message)
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg["Subject"] = topic

            try:
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, app_password)
                    server.send_message(msg)
            except Exception as e:
                print("邮件发送失败:", e)

        def date_range(start_date: datetime, end_date: datetime, delta: timedelta):
            current_date = start_date
            while current_date <= end_date:
                yield current_date
                current_date += delta

        from geo_core import GeoDownloader, GeoProcessor
        with open("sos_global_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            config_data = yaml.load(yaml_file)

        def whole_process(start_dt_str, end_dt_str):
            downloader = GeoDownloader(config_data, self.sat_index, self.product_index_list,
                                       start_dt_str[0:8], start_dt_str[8:12],
                                       end_dt_str[0:8], end_dt_str[8:12], sos=True)
            start_dt_corr_str, end_dt_corr_str = downloader.correct_date()
            self.start_date_corr.emit(start_dt_corr_str[0:8])
            self.start_time_corr.emit(start_dt_corr_str[8:12])
            self.end_date_corr.emit(end_dt_corr_str[0:8])
            self.end_time_corr.emit(end_dt_corr_str[8:12])

            (link_length_dict_dl, download_email_dict_dl,
             warning_email_dict_dl) = downloader.download_sat_files(need_download=True)

            send_mail(download_email_dict_dl["topic"], download_email_dict_dl["message"])
            if warning_email_dict_dl:
                send_mail(warning_email_dict_dl["topic"], warning_email_dict_dl["message"])

            processor = GeoProcessor(config_data, self.sat_index, self.proj_index, self.product_index_list, sos=True)
            clean_dict_geo, process_email_dict_geo, warning_email_dict_geo = processor.global_mosaic(start_dt_corr_str,
                                                                                                     end_dt_corr_str)

            send_mail(process_email_dict_geo["topic"], process_email_dict_geo["message"])
            if warning_email_dict_geo:
                send_mail(warning_email_dict_geo["topic"], warning_email_dict_geo["message"])

            return clean_dict_geo

        if self.step_mode:
            start_dt = datetime.strptime(self.start_date, "%Y%m%d%H%M")
            end_dt = datetime.strptime(self.end_date, "%Y%m%d%H%M") - timedelta(minutes=10)
            total_time_nodes = len(list(date_range(start_dt, end_dt, timedelta(minutes=10))))
            current_time_node = 0

            for tmp_dt in date_range(start_dt, end_dt, timedelta(minutes=10)):
                tmp_start_dt = tmp_dt
                tmp_end_dt = tmp_start_dt + timedelta(minutes=10)
                tmp_start_dt_str = datetime.strftime(tmp_start_dt, "%Y%m%d%H%M")
                tmp_end_dt_str = datetime.strftime(tmp_end_dt, "%Y%m%d%H%M")

                clean_dict = whole_process(tmp_start_dt_str, tmp_end_dt_str)

                current_time_node += 1

                if current_time_node == total_time_nodes:
                    clean_dict["last_round"] = True
                    self.clean_dict.emit(clean_dict)
                else:
                    clean_dict["last_round"] = False
                    self.clean_dict.emit(clean_dict)
                    with QMutexLocker(self.mutex):
                        self.condition.wait(self.mutex)

        else:
            clean_dict = whole_process(self.start_date, self.end_date)

            clean_dict["last_round"] = True
            self.clean_dict.emit(clean_dict)


class CleanThread(QThread):
    done = pyqtSignal()

    def __init__(self, sat_index, product_index_list, mutex: QMutex, condition: QWaitCondition, clean_dict=None):
        super().__init__()
        self.sat_index = sat_index
        self.product_index_list = [max(0, idx - 1) for idx in product_index_list]
        self.mutex = mutex
        self.condition = condition
        self.clean_dict = clean_dict

    def run(self):
        from geo_core import GeoProcessor
        with open("sos_global_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            config_data = yaml.load(yaml_file)

        processor = GeoProcessor(config_data, self.sat_index, 0, self.product_index_list, sos=True)
        processor.run_clean(self.clean_dict, all_clean=True)

        if self.clean_dict["last_round"] or self.clean_dict is None:
            self.done.emit()
        else:
            with QMutexLocker(self.mutex):
                self.condition.wakeAll()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UiGeo()
    ui.show()
    sys.exit(app.exec())
