import sys

from PyQt6.QtWidgets import QApplication

from QT_yolo.jiemian import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()

        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"启动失败: {str(e)}")
        sys.exit(1)