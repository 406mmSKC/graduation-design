import os
import sys
import cv2
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QGridLayout, QComboBox
)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize
from thop import profile

from ultralytics import YOLO


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 窗口基础设置
        self.setWindowTitle("YOLOv11 智能检测系统")
        self.setGeometry(200, 200, 1280, 720)
        self.setStyleSheet("""
            QMainWindow { 
                font-family: 'Segoe UI';
                /*background-image: url("code_pictures/ncepu_logo.jpg");*/
                background-repeat: no-repeat;
                background-position: center;
            }
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;  /* 减小内边距 */
                font-size: 14px;
                min-width: 100px;   /* 设置最小宽度 */
            }
            QPushButton:hover { background-color: #357ABD; }
            QLabel { 
                font-size: 14px; 
                color: #444444;
                padding: 8px;
            }
            QGroupBox {
                border: 1px solid #D3D3D3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #666666;
                background-color: rgba(255, 255, 255, 0.5);
            }
        """)

        self.source_img = ""
        self.source_video = ""
        self.result_img = ""
        self.result_video = ""
        self.base_model_path = "QT_yolo/pt/"
        self.model_path = ""

        # 主布局容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # ================= 文件选择区域 =================
        file_group = QGroupBox("文件操作")
        file_layout = QHBoxLayout()

        # 图片选择按钮
        self.img_btn = QPushButton(QIcon("image_icon.png"), " 选择图片")
        self.img_btn.setIconSize(QSize(24, 24))
        self.img_status = QLabel("图片状态：未选择")

        # 视频选择按钮
        self.video_btn = QPushButton(QIcon("video_icon.png"), " 选择视频")
        self.video_btn.setIconSize(QSize(24, 24))
        self.video_status = QLabel("视频状态：未处理")

        file_layout.addWidget(self.img_btn)
        file_layout.addWidget(self.img_status)
        file_layout.addWidget(self.video_btn)
        file_layout.addWidget(self.video_status)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # ================= 图片显示区域 =================
        img_group = QGroupBox("图像检测结果")
        img_layout = QGridLayout()
        self.img_preview = QLabel()
        self.img_result = QLabel()
        for label in [self.img_preview, self.img_result]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("""
                background: white; 
                border-radius: 8px;
                border: 2px dashed #CCCCCC;
            """)
            label.setMinimumSize(600, 400)
        img_layout.addWidget(self.img_preview, 0, 0)
        img_layout.addWidget(self.img_result, 0, 1)
        img_group.setLayout(img_layout)
        main_layout.addWidget(img_group)

        # ================= 操作按钮区域 =================
        control_group = QGroupBox("检测控制")
        btn_layout = QHBoxLayout()

        # 图片检测按钮（宽度调整）
        self.confirm_btn = QPushButton(QIcon("detect_icon.png"), " 图片检测")
        self.confirm_btn.setFixedWidth(120)  # 设置固定宽度

        # 新增视频检测按钮
        self.video_detect_btn = QPushButton(QIcon("video_detect_icon.png"), " 视频检测")
        self.video_detect_btn.setFixedWidth(120)  # 设置相同宽度

        self.save_label = QLabel("结果路径：等待检测...")

        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.video_detect_btn)  # 添加视频检测按钮
        btn_layout.addWidget(self.save_label)
        control_group.setLayout(btn_layout)
        main_layout.addWidget(control_group)

        # ================= 模型选择区域 =================
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["v11s_CCTSDB","v11s_TT100K", "v11n_CCTSDB","v11n_TT100K","v11s_CCTSDB_pro","v11s_TT100K_pro", "v11n_CCTSDB_pro","v11n_TT100K_pro"])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: white;
                border: 1px solid #D3D3D3;
                border-radius: 5px;
                padding: 5px 15px;
                min-width: 200px;
            }
            QComboBox::drop-down { width: 20px; }
        """)
        model_layout.addWidget(QLabel("当前模型："))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        main_layout.insertWidget(1, model_group)

        # 信号连接
        self.img_btn.clicked.connect(self.select_image)
        self.video_btn.clicked.connect(self.select_video)
        self.confirm_btn.clicked.connect(self.process_image)
        self.video_detect_btn.clicked.connect(self.process_video)  # 连接视频检测按钮
        self.model_combo.currentIndexChanged.connect(self.update_model_path)
    def select_image(self):
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.jpg *.png *.jpeg)", options=options
        )
        if file_path:
            self.source_img = file_path
            self.img_status.setText(f"图片状态：已选择")
            pixmap = QPixmap(file_path).scaled(
                600, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.img_preview.setPixmap(pixmap)

    def process_image(self):
        if not self.source_img:
            self.save_label.setText("错误：请先选择图片文件！")
            return

        try:
            self.save_label.setText(f"图片检测中，请不要操作！")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            print(base_dir)
            model_dir = os.path.join(base_dir, "pt")
            model_name = self.model_combo.currentText() + ".pt"
            self.model_path = os.path.join(model_dir, model_name)
            print("模型将载入权重" + self.model_path)
            model = YOLO(self.model_path)
            results = model(self.source_img, save=True, project="custom_predict_image")
            self.result_img = os.path.join(results[0].save_dir, os.path.basename(self.source_img))

            result_pixmap = QPixmap(self.result_img).scaled(
                600, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.img_result.setPixmap(result_pixmap)
            self.save_label.setText(f"结果路径：{self.result_img}")
        except Exception as e:
            self.save_label.setText(f"检测失败：{str(e)}")
    def select_video(self):
        """仅选择视频文件，不自动处理"""
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov)", options=options
        )
        if file_path:
            self.source_video = file_path
            self.video_status.setText("视频状态：已选择（等待检测）")

    def process_video(self):
        """视频检测处理核心逻辑"""
        if not self.source_video:
            self.save_label.setText("错误：请先选择视频文件！")
            return

        try:
            self.save_label.setText(f"开始处理，正在配置模型")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "pt")
            model_name = self.model_combo.currentText() + ".pt"
            self.model_path = os.path.join(model_dir, model_name)

            model = YOLO(self.model_path)
            self.save_label.setText(f"模型加载完毕，获取视频参数")
            cap = cv2.VideoCapture(self.source_video)

            # 视频参数获取
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 创建输出目录
            save_dir = os.path.join(os.path.dirname(self.source_video), "custom_predict_vedio")
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, f"pred_{os.path.basename(self.source_video)}")

            # 初始化视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 处理视频帧
            self.save_label.setText(f"视频处理中，请不要操作！")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                results = model(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            # 释放资源
            cap.release()
            out.release()

            self.result_video = output_path
            self.video_status.setText(f"视频处理完成\n保存路径：{output_path}")
            self.save_label.setText(f"最新操作：视频检测完成")

        except Exception as e:
            self.video_status.setText(f"处理失败：{str(e)}")
            self.save_label.setText(f"错误：视频检测失败")

    def update_model_path(self):
        """根据下拉选项更新模型路径"""
        # print("当前下拉框内容为"+self.model_combo.currentText())
        model_name = self.model_combo.currentText()
        self.model_path = os.path.join(
            self.base_model_path,
            f"{model_name}.pt"
        )
        print(f"已切换模型至：{self.model_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())