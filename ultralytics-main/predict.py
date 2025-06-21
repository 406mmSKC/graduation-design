from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = r"C:\Users\Lenovo\Desktop\best.pt"

    test_path = r"F:\YOLOV11\ultralytics-main\data\images\test\19010.jpg"
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=0.5)  # predict on an image
