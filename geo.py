#!/usr/bin/python
import os
import re
import shutil
import sys
import warnings
from datetime import datetime, timedelta

import pyproj
import ruamel.yaml
from PyQt6.QtCore import QDate, QRect, QRegularExpression, QThread, QTime, pyqtSignal
from PyQt6.QtGui import QFont, QIntValidator, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
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


class Button(QPushButton):
    def __init__(self, x, y, width, height, font, text, parent=None):
        super().__init__(parent)
        self.setGeometry(QRect(x, y, width, height))
        self.setFont(font)
        self.setText(text)


class UiGeo(QWidget):
    def __init__(self):
        super().__init__()
        with open("geo_config.yaml", "r", encoding="utf-8") as yaml_file:
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

        self.setWindowTitle("Geostationary Toolbox")
        self.resize(1200, 1000)

        self.tab = QTabWidget(self)
        self.tab.setGeometry(QRect(0, 0, 1200, 1000))
        self.process_tab = QWidget()
        self.download_tab = QWidget()
        self.tab.addTab(self.process_tab, "处理")
        self.tab.addTab(self.download_tab, "下载链接生成")

        self.start_button_geo = Button(830, 210, 160, 100, font, "开始", self.process_tab )
        self.start_button_geo.setEnabled(False)
        self.start_button_geo.clicked.connect(self.geo_start)

        self.label_title_geo = Label(340, 20, 740, 65, font, "在同目录下的geo_config.yaml中编辑参数并重启程序",
                                     self.process_tab)

        init_sat_list = [self.sat_dict[key]["name"] for key in self.sat_dict]
        self.cb_sat_geo = ComboBox(60, 150, 600, 40, font2, False, init_sat_list, self.process_tab)
        self.cb_sat_geo.currentIndexChanged.connect(self.sat_select_geo)
        self.cb_sat_geo.currentIndexChanged.connect(self.valid_items_geo)

        init_proj_list = [(init_proj_dict[key]["name"] + " | " + init_proj_dict[key]["area"] + " | " + "中央经线" +
                           str(init_proj_dict[key]["center_lon"]) + " | " + "分辨率" + str(init_proj_dict[key]["res"]))
                          for key in init_proj_dict]
        self.cb_proj_geo = ComboBox(60, 240, 600, 40, font2, False, init_proj_list, self.process_tab)
        self.cb_proj_geo.currentIndexChanged.connect(self.area_select)

        init_comp_list = [init_comp_dict[key]["name"] for key in init_comp_dict]
        self.cb_product_geo = ComboCheckBox(60, 330, 600, 40, font2, init_comp_list, parent=self.process_tab)
        self.cb_product_geo.ListChecked.connect(self.valid_items_geo)

        self.label_sat_geo = Label(60, 110, 200, 40, font1, "选择卫星", self.process_tab)
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

        self.lineEdit_latlon_calc = LineEdit(280, 690, 520, 40, font1, self.process_tab)
        self.lineEdit_proj4_calc = LineEdit(60, 780, 580, 40, font1, self.process_tab)
        self.lineEdit_name_calc = LineEdit(660, 780, 150, 40, font1, self.process_tab)
        self.lineEdit_area_calc = LineEdit(820, 780, 150, 40, font1, self.process_tab)
        self.lineEdit_area_id_calc = LineEdit(980, 780, 150, 40, font1, self.process_tab)

        self.lineEdit_latlon_calc.textChanged.connect(self.valid_proj_enable_calc)
        self.lineEdit_proj4_calc.textChanged.connect(self.valid_proj_enable_calc)
        self.lineEdit_name_calc.textChanged.connect(self.valid_proj_enable_calc)
        self.lineEdit_area_calc.textChanged.connect(self.valid_proj_enable_calc)
        self.lineEdit_area_id_calc.textChanged.connect(self.valid_proj_enable_calc)

        int_validator = QIntValidator()

        regex_1 = QRegularExpression("[-0-9. ]*")
        reg_validator = QRegularExpressionValidator(regex_1)
        self.lineEdit_latlon_calc.setValidator(reg_validator)

        regex_2 = QRegularExpression("^[\u4e00-\u9fa5]*$")
        chs_validator = QRegularExpressionValidator(regex_2)
        self.lineEdit_name_calc.setValidator(chs_validator)

        regex_3 = QRegularExpression("^[\u4e00-\u9fa5（）()0-9]*$")
        chs_validator_2 = QRegularExpressionValidator(regex_3)
        self.lineEdit_area_calc.setValidator(chs_validator_2)

        regex_4 = QRegularExpression("^[a-zA-Z]*$")
        eng_validator = QRegularExpressionValidator(regex_4)
        self.lineEdit_area_id_calc.setValidator(eng_validator)

        regex_5 = QRegularExpression("^[a-zA-Z0-9+=_ -]*$")
        eng_validator_2 = QRegularExpressionValidator(regex_5)
        self.lineEdit_proj4_calc.setValidator(eng_validator_2)

        self.label_latlon_calc = Label(280, 650, 520, 40, font2,
                                       "小数点经纬度四至范围 (min_lon  min_lat  max_lon  max_lat)", self.process_tab)
        self.label_proj4_calc = Label(60, 740, 1070, 40, font2, "Proj4表达式", self.process_tab)
        self.label_name_calc = Label(660, 740, 150, 40, font2, "地图投影(中文)", self.process_tab)
        self.label_area_calc = Label(820, 740, 150, 40, font2, "区域(中文)", self.process_tab)
        self.label_area_id_calc = Label(980, 740, 150, 40, font2, "区域代码(英文)", self.process_tab)

        self.start_button_calc = Button(850, 690, 280, 41, font2, "计算宽高xy并写入刷新yaml", self.process_tab)
        self.start_button_calc.clicked.connect(self.calc_start)
        self.start_button_calc.setEnabled(False)

        self.geo_thread = GeoThread(0, 0, [1])
        self.geo_thread.done.connect(self.geo_done_event)

        self.clean_thread = CleanThread(0, [1])
        self.clean_thread.done.connect(self.clean_done_event)

        self.calc_thread = CalcThread(0, "", "", "", "", "")
        self.calc_thread.done.connect(self.calc_done_event)

        self.label_title_dl = Label(340, 20, 740, 65, font, "需要联网", self.download_tab)

        self.start_button_dl = Button(830, 150, 160, 100, font, "生成下载链接", self.download_tab)
        self.start_button_dl.setEnabled(False)

        self.label_sat_dl = Label(60, 110, 200, 40, font1, "选择卫星", self.download_tab)
        self.label_status_dl = Label(660, 350, 400, 200, font, "", self.download_tab)

        self.cb_sat_dl = ComboBox(60, 150, 600, 40, font2, False, init_sat_list, self.download_tab)
        self.cb_sat_dl.currentIndexChanged.connect(self.sat_select_dl)

        self.label_product_dl = Label(60, 200, 200, 40, font1, "选择产品", self.download_tab)

        self.label_start_date_dl = Label(60, 360, 250, 40, font2, "观测起始日期8位数字", self.download_tab)
        self.label_start_time_dl = Label(330, 360, 190, 40, font2, "观测起始时间4位数字", self.download_tab)
        self.label_end_date_dl = Label(60, 460, 250, 40, font2, "观测结束日期8位数字", self.download_tab)
        self.label_end_time_dl = Label(330, 460, 190, 40, font2, "观测结束时间4位数字", self.download_tab)

        self.lineEdit_start_date_dl = LineEdit(60, 400, 250, 40, font1, self.download_tab)
        self.lineEdit_start_time_dl = LineEdit(330, 400, 190, 40, font1, self.download_tab)
        self.lineEdit_end_date_dl = LineEdit(60, 500, 250, 40, font1, self.download_tab)
        self.lineEdit_end_time_dl = LineEdit(330, 500, 190, 40, font1, self.download_tab)

        self.lineEdit_start_date_dl.setMaxLength(8)
        self.lineEdit_start_time_dl.setMaxLength(4)
        self.lineEdit_end_date_dl.setMaxLength(8)
        self.lineEdit_end_time_dl.setMaxLength(4)

        self.lineEdit_start_date_dl.setValidator(int_validator)
        self.lineEdit_start_time_dl.setValidator(int_validator)
        self.lineEdit_end_date_dl.setValidator(int_validator)
        self.lineEdit_end_time_dl.setValidator(int_validator)

        self.lineEdit_start_date_dl.textChanged.connect(self.valid_dt_and_items_dl)
        self.lineEdit_start_time_dl.textChanged.connect(self.valid_dt_and_items_dl)
        self.lineEdit_end_date_dl.textChanged.connect(self.valid_dt_and_items_dl)
        self.lineEdit_end_time_dl.textChanged.connect(self.valid_dt_and_items_dl)

        self.cb_product_dl = ComboCheckBox(60, 240, 600, 40, font2, init_comp_list, parent=self.download_tab)
        self.cb_product_dl.ListChecked.connect(self.valid_dt_and_items_dl)

        self.lineEdit_start_date_dl.setText(datetime(datetime.now().year, 1, 1).strftime("%Y%m%d"))
        self.lineEdit_start_time_dl.setText("0000")
        self.lineEdit_end_date_dl.setText(datetime(datetime.now().year, 1, 1).strftime("%Y%m%d"))
        self.lineEdit_end_time_dl.setText("0010")

        self.dl_thread = DownloadThread("20230101", "0150", "20230101", "0150", 0, [1])

        self.start_button_dl.clicked.connect(self.dl_start)

    def valid_items_geo(self):
        ret, ret_idx = self.cb_product_geo.get_selected()
        self.start_button_geo.setEnabled(bool(ret_idx))

    def valid_dt_and_items_dl(self):
        ret, ret_idx = self.cb_product_dl.get_selected()
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

        self.start_button_dl.setEnabled(all([valid_items, valid_length, valid_date, valid_range]))

    def sat_select_geo(self):
        idx = self.cb_sat_geo.currentIndex()
        self.cb_proj_geo.clear()

        satellite = {"GOES-WEST": "GOES-WEST",
                     "GOES-EAST": "GOES-WEST",
                     "MTG-FD": "MTG-FD",
                     "MTG-EU": "MTG-FD"}.get(list(self.sat_dict)[idx], list(self.sat_dict)[idx])

        proj_dict = self.config_data["projection"][list(self.sat_dict)[idx]]
        proj_list = [(proj_dict[key]["name"] + " | " + proj_dict[key]["area"] + " | " + "中央经线" +
                      str(proj_dict[key]["center_lon"]) + " | " + "分辨率" + str(proj_dict[key]["res"]))
                     for key in proj_dict]
        for proj in proj_list:
            self.cb_proj_geo.addItem(proj)

        comp_dict = self.config_data["composites"][satellite]
        comp_list = [comp_dict[key]["name"] for key in comp_dict]
        self.cb_product_geo.new_list(comp_list)

    def area_select(self):
        idx = self.cb_sat_geo.currentIndex()
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
        self.cb_sat_geo.setEnabled(False)
        self.cb_proj_geo.setEnabled(False)
        self.cb_product_geo.setEnabled(False)

        sat_index = self.cb_sat_geo.currentIndex()
        proj_index = self.cb_proj_geo.currentIndex()
        product_index_list = self.cb_product_geo.get_selected()[1]

        self.geo_thread = GeoThread(sat_index, proj_index, product_index_list)
        self.geo_thread.done.connect(self.geo_done_event)
        self.geo_thread.start()

    def geo_done_event(self):
        self.geo_thread.quit()

        sat_index = self.cb_sat_geo.currentIndex()
        product_index_list = self.cb_product_geo.get_selected()[1]

        self.clean_thread = CleanThread(sat_index, product_index_list)
        self.clean_thread.done.connect(self.clean_done_event)
        self.clean_thread.start()

    def clean_done_event(self):
        self.clean_thread.quit()
        self.start_button_geo.setEnabled(True)
        self.cb_sat_geo.setEnabled(True)
        self.cb_proj_geo.setEnabled(True)
        self.cb_product_geo.setEnabled(True)

    def valid_proj_enable_calc(self):
        pattern = re.compile(r"^-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+-?\d+(\.\d+)?$")
        if pattern.match(self.lineEdit_latlon_calc.text()):
            latlon_values = [float(part) for part in self.lineEdit_latlon_calc.text().split()]
            if ((-180 <= latlon_values[0] <= 180 and -180 <= latlon_values[2] <= 180 and
                 -90 <= latlon_values[1] <= 90 and -90 <= latlon_values[3] <= 90) and
                    (latlon_values[0] < latlon_values[2] and latlon_values[1] < latlon_values[3])):
                latlon_pass = True
            else:
                latlon_pass = False
        else:
            latlon_pass = False

        name_pass = True if len(self.lineEdit_name_calc.text()) > 0 else False
        area_pass = True if len(self.lineEdit_area_calc.text()) > 0 else False
        area_id_pass = True if len(self.lineEdit_area_id_calc.text()) > 0 else False
        proj4_pass = True if check_proj4_validity(self.lineEdit_proj4_calc.text()) else False

        self.start_button_calc.setEnabled(all([latlon_pass, name_pass, area_pass, area_id_pass, proj4_pass]))

    def calc_start(self):
        self.start_button_calc.setEnabled(False)
        self.lineEdit_latlon_calc.setEnabled(False)
        self.lineEdit_proj4_calc.setEnabled(False)
        self.lineEdit_area_calc.setEnabled(False)
        self.lineEdit_area_id_calc.setEnabled(False)
        self.lineEdit_name_calc.setEnabled(False)

        self.cb_sat_geo.setEnabled(False)
        self.cb_proj_geo.setEnabled(False)
        self.cb_product_geo.setEnabled(False)

        sat_index = self.cb_sat_geo.currentIndex()
        calc_latlon = self.lineEdit_latlon_calc.text()
        calc_proj = self.lineEdit_proj4_calc.text()
        calc_name = self.lineEdit_name_calc.text()
        calc_area = self.lineEdit_area_calc.text()
        calc_area_id = self.lineEdit_area_id_calc.text()
        self.calc_thread = CalcThread(sat_index, calc_latlon, calc_proj, calc_name, calc_area, calc_area_id)
        self.calc_thread.done.connect(self.calc_done_event)

        self.calc_thread.start()

    def calc_done_event(self):
        self.calc_thread.quit()

        self.cb_proj_geo.clear()

        with open("geo_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            self.config_data = yaml.load(yaml_file)

        proj_dict = self.config_data["projection"][list(self.sat_dict)[self.cb_sat_geo.currentIndex()]]
        for key in proj_dict:
            name = proj_dict[key]["name"]
            area = proj_dict[key]["area"]
            center_lon = proj_dict[key]["center_lon"]
            res = proj_dict[key]["res"]
            description = name + " | " + area + " | " + "中央经线" + str(center_lon) + " | " + "分辨率" + str(res)
            self.cb_proj_geo.addItem(description)

        self.start_button_calc.setEnabled(True)
        self.lineEdit_latlon_calc.setEnabled(True)
        self.lineEdit_proj4_calc.setEnabled(True)
        self.lineEdit_area_calc.setEnabled(True)
        self.lineEdit_area_id_calc.setEnabled(True)
        self.lineEdit_name_calc.setEnabled(True)

        self.cb_sat_geo.setEnabled(True)
        self.cb_proj_geo.setEnabled(True)
        self.cb_product_geo.setEnabled(True)

    def sat_select_dl(self):
        idx = self.cb_sat_dl.currentIndex()

        satellite = {"GOES-WEST": "GOES-WEST",
                     "GOES-EAST": "GOES-WEST",
                     "MTG-FD": "MTG-FD",
                     "MTG-EU": "MTG-FD"}.get(list(self.sat_dict)[idx], list(self.sat_dict)[idx])

        comp_dict = self.config_data["composites"][satellite]
        comp_list = [comp_dict[key]["name"] for key in comp_dict]
        self.cb_product_dl.new_list(comp_list)

    def dl_valid_date(self, text):
        try:
            date = QDate.fromString(text, "yyyyMMdd")
            self.start_button_dl.setEnabled(date.isValid())
        except ValueError:
            self.start_button_dl.setEnabled(False)

    def dl_valid_time(self, text):
        try:
            time = QTime.fromString(text, "HHmm")
            self.start_button_dl.setEnabled(time.isValid())
        except ValueError:
            self.start_button_dl.setEnabled(False)

    def dl_start(self):
        self.start_button_dl.setEnabled(False)
        self.label_status_dl.setText("")
        self.lineEdit_start_date_dl.setEnabled(False)
        self.lineEdit_start_time_dl.setEnabled(False)
        self.lineEdit_end_date_dl.setEnabled(False)
        self.lineEdit_end_time_dl.setEnabled(False)

        dl_start_date = self.lineEdit_start_date_dl.text()
        dl_start_time = self.lineEdit_start_time_dl.text()
        dl_end_date = self.lineEdit_end_date_dl.text()
        dl_end_time = self.lineEdit_end_time_dl.text()
        dl_sat_index = self.cb_sat_dl.currentIndex()
        dl_product_index_list = self.cb_product_dl.get_selected()[1]

        self.dl_thread = DownloadThread(dl_start_date, dl_start_time, dl_end_date, dl_end_time,
                                        dl_sat_index, dl_product_index_list)
        self.dl_thread.date_order_error.connect(self.dl_date_order_error)
        self.dl_thread.start_date_corr.connect(self.start_date_corr)
        self.dl_thread.start_time_corr.connect(self.start_time_corr)
        self.dl_thread.end_date_corr.connect(self.end_date_corr)
        self.dl_thread.end_time_corr.connect(self.end_time_corr)
        self.dl_thread.nothing.connect(self.dl_nothing)
        self.dl_thread.done.connect(self.dl_done_event)

        self.dl_thread.start()

    def dl_done_event(self):
        self.dl_thread.quit()
        self.label_status_dl.setText(""
                                     "已生成txt\n"
                                     "2015年11月1日前数据请前往dias-jp官网\n"
                                     "2019年8月1日前数据请前往KMA官网")
        self.start_button_dl.setEnabled(True)
        self.lineEdit_start_date_dl.setEnabled(True)
        self.lineEdit_start_time_dl.setEnabled(True)
        self.lineEdit_end_date_dl.setEnabled(True)
        self.lineEdit_end_time_dl.setEnabled(True)

    def dl_date_order_error(self):
        self.label_status_dl.setText("结束日期应至少晚于起始日期10分钟")
        self.start_button_dl.setEnabled(True)
        self.lineEdit_start_date_dl.setEnabled(True)
        self.lineEdit_start_time_dl.setEnabled(True)
        self.lineEdit_end_date_dl.setEnabled(True)
        self.lineEdit_end_time_dl.setEnabled(True)

    def dl_nothing(self):
        self.label_status_dl.setText("此段时间无数据/无需额外下载")
        self.start_button_dl.setEnabled(True)
        self.lineEdit_start_date_dl.setEnabled(True)
        self.lineEdit_start_time_dl.setEnabled(True)
        self.lineEdit_end_date_dl.setEnabled(True)
        self.lineEdit_end_time_dl.setEnabled(True)

    def start_date_corr(self, string):
        self.lineEdit_start_date_dl.setText(string)

    def start_time_corr(self, string):
        self.lineEdit_start_time_dl.setText(string)

    def end_date_corr(self, string):
        self.lineEdit_end_date_dl.setText(string)

    def end_time_corr(self, string):
        self.lineEdit_end_time_dl.setText(string)


class GeoThread(QThread):
    geos_proj_corr = pyqtSignal()
    done = pyqtSignal()

    def __init__(self, sat_index, proj_index, product_index_list):
        super(GeoThread, self).__init__()
        self.sat_index = sat_index
        self.proj_index = proj_index
        self.product_index_list = [max(0, idx - 1) for idx in product_index_list]

    def run(self):
        with open("geo_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            config_data = yaml.load(yaml_file)

        conda_path = config_data["basic_path"]["conda_path"]
        satpy_config_path = config_data["basic_path"]["satpy_config_path"]
        base_path = config_data["basic_path"]["base_path"]
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        geo_float_image_reader_script = os.path.join(satpy_config_path, "modified_script/geo_float_image.py")
        geo_float_image_reader = os.path.join(conda_path, "Lib/site-packages/satpy/readers/geo_float_image.py")
        if not os.path.exists(geo_float_image_reader):
            shutil.copy(geo_float_image_reader_script, geo_float_image_reader)

        from geo_core import GeoProcessor
        processor = GeoProcessor(config_data, self.sat_index, self.proj_index, self.product_index_list)
        processor.run_satpy_processor()

        self.done.emit()


class CleanThread(QThread):
    done = pyqtSignal()

    def __init__(self, sat_index, product_index_list):
        super().__init__()
        self.sat_index = sat_index
        self.product_index_list = [max(0, idx - 1) for idx in product_index_list]

    def run(self):
        from geo_core import GeoProcessor
        with open("geo_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            config_data = yaml.load(yaml_file)

        processor = GeoProcessor(config_data, self.sat_index, 0, self.product_index_list)
        processor.run_clean()

        self.done.emit()


class CalcThread(QThread):
    done = pyqtSignal()

    def __init__(self, sat_index, calc_latlon, calc_proj, calc_name, calc_area, calc_area_id):
        super().__init__()
        self.sat_index = sat_index
        self.calc_latlon = calc_latlon
        self.calc_proj = calc_proj
        self.calc_name = calc_name
        self.calc_area = calc_area
        self.calc_area_id = calc_area_id

    def run(self):
        from geo_core import GeoCalculator
        with open("geo_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            config_data = yaml.load(yaml_file)

        calculator = GeoCalculator(config_data, self.sat_index, self.calc_latlon, self.calc_proj, self.calc_name,
                                   self.calc_area, self.calc_area_id)

        new_config_data = calculator.projection_calc()

        with open("geo_config.yaml", "w", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            yaml.dump(new_config_data, yaml_file)

        self.done.emit()


class DownloadThread(QThread):
    date_order_error = pyqtSignal()
    done = pyqtSignal()
    start_date_corr = pyqtSignal(str)
    start_time_corr = pyqtSignal(str)
    end_date_corr = pyqtSignal(str)
    end_time_corr = pyqtSignal(str)
    nothing = pyqtSignal()

    def __init__(self, start_date, start_time, end_date, end_time, sat_index, product_index_list):
        super().__init__()
        self.start_date = start_date
        self.start_time = start_time
        self.end_date = end_date
        self.end_time = end_time
        self.sat_index = sat_index
        self.product_index_list = [max(0, idx - 1) for idx in product_index_list]

    def run(self):
        from geo_core import GeoDownloader
        with open("geo_config.yaml", "r", encoding="utf-8") as yaml_file:
            yaml = ruamel.yaml.YAML(typ="rt")
            config_data = yaml.load(yaml_file)

        base_path = config_data["basic_path"]["base_path"]
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        downloader = GeoDownloader(config_data, self.sat_index, self.product_index_list,
                                   self.start_date, self.start_time, self.end_date, self.end_time)
        start_dt, end_dt = downloader.correct_date()
        self.start_date_corr.emit(start_dt[0:8])
        self.start_time_corr.emit(start_dt[8:12])
        self.end_date_corr.emit(end_dt[0:8])
        self.end_time_corr.emit(end_dt[8:12])

        link_length_dict = downloader.download_sat_files()[0]

        for sat, link_length in link_length_dict.items():
            if link_length == 0:
                self.nothing.emit()
            else:
                self.done.emit()
            break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UiGeo()
    ui.show()
    sys.exit(app.exec())
