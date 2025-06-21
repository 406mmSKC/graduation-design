import warnings

from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/12/yolo12n.yaml')  # 修改yaml
    model.load('yolo12n.pt')
    model.train(data='/tmp/pycharm_project_538/data.yaml',
                imgsz=640,
                epochs=300,
                batch=200,
                workers=25,
                device=[0,1],
                optimizer='SGD',
                amp=True,
                cache=False,
                #resume=True
                )