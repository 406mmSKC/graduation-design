import warnings

from ultralytics import YOLO
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11s.yaml')  # 修改yaml
    model.load('yolo11s.pt')
    model.train(data='/root/autodl-tmp/.autodl/tt100k_yolo/tt100k.yaml',
                imgsz=640,
                epochs=300,
                batch=132,
                workers=35,
                device=[0,1,2],
                optimizer='SGD',
                amp=True,
                cache=False,
                #resume=True
                )