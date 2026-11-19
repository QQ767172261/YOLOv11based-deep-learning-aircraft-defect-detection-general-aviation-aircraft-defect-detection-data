通航飞机缺陷检测数据集 yolo检测模型 通航飞机缺陷检测系统界面![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4d43db37132e4fe9b3ba6c7efac4c422.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f9ded3c9910145439f04b18b3fd67876.png)

通航飞机缺陷检测数据集，4169张，提供yolo，voc，coco三种标注方式
图像尺寸:640*640![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/22e2e35e6bf14ec28042d116dbc4b9b4.png)

类别数量:2类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8b122e20efce44a5bbc5439cb264a257.png)

训练集图像数量:3659; 验证集图像数量:416； 测试集图像数量:94
类别名称: 每一类图像数 ，每一类标注数
裂缝 2913,5970
凹痕 1631,3791
image num: 4169


模型与源代码
采用 YOLOv11n 网络训练
训练轮次：80 个 epoch![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/194595d63f3c4b8689ca2d08c20458c8.png)

提供训练好的yolo11n模型，全部训练 + 测试源代码
训练精度 mAP 效果如图所示
PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1633e2c9e04e437d8c9b01e066e5d16a.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测
界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/967cf2b88be64b58b601c9dd14630bf8.png)

操作简单直观，无需命令行
4、
运行环境：Python=3.8、opencv-python、PyQt5、torch
支持 Windows、Linux 系统、
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83d1d128cd7f48fe89ce893b9caa514f.png)


核对完成：4169 = 3659 + 416 + 94，总标注框 9761 个，平均每图 2.34 个；PR 图数值：crack 0.541、dent 0.492、全类 mAP@0.5 = 0.517。

### 表 1 数据集概览

| 项目 | 内容 |
| --- | --- |
| 数据集名称 | 通航飞机缺陷检测数据集 |
| 图像总数 | 4169 张 |
| 图像尺寸 | 640 × 640 |
| 任务类型 | 目标检测（缺陷定位 + 类型分类） |
| 类别数 | 2 类（裂缝、凹痕） |
| 标注格式 | YOLO / VOC / COCO 三种同源标注 |
| 总标注数 | 9761 个，平均每图 2.34 个 |
| 数据划分 | train 3659 / val 416 / test 94 |
| 训练基线 | YOLOv11n，80 epochs（权重已提供） |
| 配套界面 | PyQt5 完整源码（.ui / .qrc / .py + 图标），图片 / 视频 / 摄像头 |
| 运行环境 | Python 3.8 + opencv-python + PyQt5 + torch，Windows / Linux |

### 表 2 数据划分

| 子集 | 图像数 | 占比 |
| --- | --- | --- |
| 训练集 train | 3659 | 87.8% |
| 验证集 val | 416 | 10.0% |
| 测试集 test | 94 | 2.3% |
| 合计 | 4169 | 100% |

### 表 3 类别分布

| 序号 | 类名 | 中文 | 含该类图像数 | 标注数 | 图像覆盖率 | 平均标注/图 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | crack | 裂缝 | 2913 | 5970 | 69.9% | 2.05 |
| 1 | dent | 凹痕 | 1631 | 3791 | 39.1% | 2.32 |
| — | 合计 | — | — | 9761 | — | 2.34 |

### 表 4 训练指标（来自 PR 曲线）

| 类别 | AP@0.5 |
| --- | --- |
| crack 裂缝 | 0.541 |
| dent 凹痕 | 0.492 |
| 全类 mAP@0.5 | 0.517 |

两类都在 0.5 上下，属“能跑通但不精”的水平。凹痕低于裂缝，因为凹痕靠光影和细微形变判断，在 640 分辨率下特征很弱。

### 表 5 标签配置（data.yaml）

```yaml
path: ./datasets/aircraft_defect
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: crack
  1: dent
```

### 表 6 界面模块对照

| 界面区块 | 内容 | 功能 |
| --- | --- | --- |
| 文件导入 | 图片路径、视频路径、摄像头状态 | 三个入口切换检测源 |
| 检测结果 | 用时、目标数目、目标选择下拉、置信度、目标位置 xmin/ymin/xmax/ymax | 实时显示当前选中目标信息 |
| 检测结果与位置信息 | 序号 / 文件路径 / 类别 / 置信度 / 坐标位置 | 全目标列表 |
| 操作 | 保存、退出 | 导出结果图、关闭程序 |

### 表 7 简易代码

**训练 `train.py`（YOLOv11n，80 epoch）**

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data="data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,               # 显存不足改 8
        device=0,               # 无 GPU 写 "cpu"
        patience=20,
        cos_lr=True,
        scale=0.6,
        hsv_v=0.4,              # 凹痕靠光影，亮度增强要开
        mosaic=1.0,
        project="runs/detect",
        name="aircraft_defect_yolo11n",
    )
    m = model.val(data="data.yaml", split="val")
    print("mAP50:", m.box.map50, "mAP50-95:", m.box.map)
```

**检测端 `detector.py`**

```python
import cv2
from ultralytics import YOLO


class Detector:
    def __init__(self, weights="runs/detect/aircraft_defect_yolo11n/weights/best.pt",
                 conf=0.35, iou=0.6):
        self.model = YOLO(weights)
        self.conf, self.iou = conf, iou

    def infer(self, frame):
        res = self.model(frame, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)[0]
        vis = res.plot(line_width=2)
        items = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            items.append({
                "cls": self.model.names[int(box.cls[0])],
                "conf": round(float(box.conf[0]), 4),
                "box": (x1, y1, x2, y2),
            })
        return vis, items, res.speed.get("inference", 0)


if __name__ == "__main__":
    det = Detector()
    vis, items, ms = det.infer(cv2.imread("test.jpg"))
    print(len(items), f"{ms:.1f}ms", items[:3])
    cv2.imwrite("result.jpg", vis)
```

**PyQt5 界面 `main.py`（按截图布局复刻）**

```python
import sys, os, csv
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QTableWidget,
                             QTableWidgetItem, QFileDialog, QComboBox, QGroupBox,
                             QLineEdit, QHeaderView)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from detector import Detector


class AircraftDefectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的通航飞机缺陷检测系统")
        self.resize(1250, 840)
        self.det = Detector()
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.cur_vis = None
        self.items = []
        self.src_name = ""

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        title = QLabel("基于YOLOv11的飞机缺陷类型检测")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:24px; font-weight:bold; padding:10px;")
        root.addWidget(title)

        body = QHBoxLayout()
        # ---- 左：图像 ----
        self.view = QLabel("等待导入图片 / 视频")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(780, 470)
        self.view.setStyleSheet("border:1px solid #999; background:#fafafa;")
        body.addWidget(self.view, 3)

        # ---- 右：文件导入 / 检测结果 / 操作 ----
        right = QVBoxLayout()

        g_file = QGroupBox("文件导入")
        vf = QVBoxLayout(g_file)
        self.le_img = QLineEdit(); self.le_img.setPlaceholderText("请选择图片文件")
        btn_img = QPushButton("导入图片")
        self.le_vid = QLineEdit(); self.le_vid.setPlaceholderText("请选择视频文件")
        btn_vid = QPushButton("导入视频")
        self.le_cam = QLineEdit("摄像头未开启"); self.le_cam.setReadOnly(True)
        btn_cam = QPushButton("开启摄像头")
        for w in (self.le_img, btn_img, self.le_vid, btn_vid, self.le_cam, btn_cam):
            vf.addWidget(w)

        g_res = QGroupBox("检测结果")
        gr = QGridLayout(g_res)
        gr.addWidget(QLabel("用时："), 0, 0)
        self.lb_time = QLabel("0.000 s"); gr.addWidget(self.lb_time, 0, 1)
        gr.addWidget(QLabel("目标数目："), 1, 0)
        self.lb_num = QLabel("0"); gr.addWidget(self.lb_num, 1, 1)
        gr.addWidget(QLabel("目标选择："), 2, 0)
        self.cb_target = QComboBox(); self.cb_target.addItem("全部")
        self.cb_target.currentIndexChanged.connect(self.on_select)
        gr.addWidget(self.cb_target, 2, 1)
        gr.addWidget(QLabel("置信度："), 3, 0)
        self.lb_conf = QLabel("-"); self.lb_conf.setStyleSheet("color:#c0392b; font-weight:bold")
        gr.addWidget(self.lb_conf, 3, 1)
        gr.addWidget(QLabel("目标位置"), 4, 0, 1, 2)
        gr.addWidget(QLabel("xmin:"), 5, 0)
        self.lb_xmin = QLabel("-"); gr.addWidget(self.lb_xmin, 5, 1)
        gr.addWidget(QLabel("ymin:"), 6, 0)
        self.lb_ymin = QLabel("-"); gr.addWidget(self.lb_ymin, 6, 1)
        gr.addWidget(QLabel("xmax:"), 7, 0)
        self.lb_xmax = QLabel("-"); gr.addWidget(self.lb_xmax, 7, 1)
        gr.addWidget(QLabel("ymax:"), 8, 0)
        self.lb_ymax = QLabel("-"); gr.addWidget(self.lb_ymax, 8, 1)

        g_op = QGroupBox("操作")
        vo = QHBoxLayout(g_op)
        btn_save = QPushButton("保存")
        btn_csv = QPushButton("导出CSV")
        btn_exit = QPushButton("退出")
        for b in (btn_save, btn_csv, btn_exit):
            vo.addWidget(b)

        right.addWidget(g_file)
        right.addWidget(g_res)
        right.addWidget(g_op)
        right.addStretch(1)
        body.addLayout(right, 1)
        root.addLayout(body, 5)

        # ---- 下：结果表 ----
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["序号", "文件路径", "类别", "置信度", "坐标位置"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.table, 2)

        btn_img.clicked.connect(self.open_image)
        btn_vid.clicked.connect(self.open_video)
        btn_cam.clicked.connect(self.open_camera)
        btn_save.clicked.connect(self.save_image)
        btn_csv.clicked.connect(self.save_csv)
        btn_exit.clicked.connect(self.close)

    # ---------- 推理 ----------
    def run_frame(self, frame):
        vis, items, ms = self.det.infer(frame)
        self.cur_vis, self.items = vis, items
        self.show_cv(vis)
        self.lb_time.setText(f"{ms/1000:.3f} s")
        self.lb_num.setText(str(len(items)))
        self.fill_table()
        self.fill_combo()

    def fill_table(self):
        self.table.setRowCount(0)
        for i, it in enumerate(self.items):
            r = self.table.rowCount()
            self.table.insertRow(r)
            x1, y1, x2, y2 = it["box"]
            vals = [str(i + 1), self.src_name, it["cls"],
                    f"{it['conf']*100:.1f}%", f"[{x1}, {y1}, {x2}, {y2}]"]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))

    def fill_combo(self):
        self.cb_target.blockSignals(True)
        self.cb_target.clear()
        self.cb_target.addItem("全部")
        for i, it in enumerate(self.items):
            self.cb_target.addItem(f"目标{i+1} - {it['cls']}")
        self.cb_target.blockSignals(False)
        if self.items:
            self.cb_target.setCurrentIndex(1)
            self.on_select(0)
        else:
            for lb in (self.lb_xmin, self.lb_ymin, self.lb_xmax, self.lb_ymax):
                lb.setText("-")
            self.lb_conf.setText("-")

    def on_select(self, idx):
        if not self.items or idx <= 0:
            return
        it = self.items[idx - 1]
        x1, y1, x2, y2 = it["box"]
        self.lb_conf.setText(f"{it['conf']*100:.2f} %")
        self.lb_xmin.setText(str(x1)); self.lb_ymin.setText(str(y1))
        self.lb_xmax.setText(str(x2)); self.lb_ymax.setText(str(y2))
        self.table.selectRow(idx - 1)

    # ---------- 图片 ----------
    def open_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        if not p:
            return
        self.stop()
        self.src_name = os.path.basename(p)
        self.le_img.setText(p)
        self.run_frame(cv2.imread(p))

    # ---------- 视频 ----------
    def open_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if not p:
            return
        self.le_vid.setText(p)
        self.src_name = os.path.basename(p)
        self.start(p)

    # ---------- 摄像头 ----------
    def open_camera(self):
        self.src_name = "camera-0"
        self.le_cam.setText("摄像头已开启")
        self.start(0)

    def start(self, source):
        self.stop()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.lb_time.setText("无法打开视频源")
            return
        self.timer.start(1)

    def next_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return
        self.run_frame(frame)

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.le_cam.setText("摄像头未开启")

    # ---------- 显示 / 导出 ----------
    def show_cv(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(img).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_image(self):
        if self.cur_vis is None:
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存结果", "result.jpg", "JPG (*.jpg)")
        if p:
            cv2.imwrite(p, self.cur_vis)

    def save_csv(self):
        if not self.items:
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出结果", "detect.csv", "CSV (*.csv)")
        if not p:
            return
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["序号", "文件路径", "类别", "置信度", "xmin", "ymin", "xmax", "ymax"])
            for i, it in enumerate(self.items, 1):
                w.writerow([i, self.src_name, it["cls"], it["conf"], *it["box"]])

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AircraftDefectUI()
    win.show()
    sys.exit(app.exec_())
```

```bash
pip install ultralytics opencv-python pyqt5
python train.py     # 训练 80 epoch
python main.py      # 启动界面
```

两点优化建议：一是 mAP 只有 0.517，根因在 640 输入下缺陷太小，可试 `imgsz=960` 或把图像切块训练，通常能提 10–15 个点；二是 test 集只有 94 张，结论波动大，汇报时用 val 416 张做主要指标，test 仅作参考。
