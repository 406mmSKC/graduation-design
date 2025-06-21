import warnings

from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('runs/detect/train2/weights/last.pt')  # 修改yaml
    model.train(data='/tmp/pycharm_project_538/data.yaml',
                imgsz=640,
                epochs=302,
                batch=140,
                workers=8,
                device=0,
                optimizer='SGD',
                amp=True,
                cache=False,
                resume=True
                )