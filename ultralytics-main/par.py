import torch
from ultralytics import YOLO
model = YOLO("C:/Users/Lenovo/Desktop/YOLOv11nTT100K300轮/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("YOLOv11nTT100K300轮总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sTT100K300轮/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sTT100K300轮总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11nCCTSDB300轮/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11nCCTSDB300轮总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sCCTSDB300/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sCCTSDB300轮总参数量:", total_params)

print()
#------------------------------------------------------------------------------
model = YOLO("C:/Users/Lenovo/Desktop/yolov12nTT100K300轮/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov12nTT100K300轮总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov12sTT100K300轮/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov12sTT100K300轮总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov12nCCTSDB300轮/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov12nCCTSDB300轮总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov12sCCTSDB 84轮（提前终止/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov12sCCTSDB 84轮（提前终止 总参数量:", total_params)

print()
#--------------------------------------------------------------------

model = YOLO("C:/Users/Lenovo/Desktop/YOLOV11nTT100K300轮删除P5/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("YOLOV11nTT100K300轮删除P5总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sTT100K300轮删除p5/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sTT100K300轮删除p5总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11nCCSTDB300轮删除P5/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11nCCSTDB300轮删除P5总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/YOLOv11sCCSTDB300轮删除p5/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("YOLOv11sCCSTDB300轮删除p5总参数量:", total_params)

print()
#--------------------------------------------------------------------

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sTT100K300轮删除P5增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sTT100K300轮删除P5增加P2 总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11nTT100K300轮删除P5增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11nTT100K300轮删除P5增加P2 总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sCCSTDB 90轮提前停止 删除P5增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sCCSTDB 90轮提前停止 删除P5增加P2 总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11nCCTSDB300轮删除P5添加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11nCCTSDB300轮删除P5添加P2 总参数量:", total_params)

print()
#--------------------------------------------------------------------

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sTT100K300轮增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sTT100K300轮增加P2 总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11nTT100K300轮增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11nTT100K300轮增加P2 总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11sCCTSDB提前停止80轮增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11sCCTSDB提前停止80轮增加P2 总参数量:", total_params)

model = YOLO("C:/Users/Lenovo/Desktop/yolov11nCCSDB提前停止103轮增加P2/weights/best.pt")
p = model.state_dict()
total_params = sum(p.numel() for p in model.parameters())
print("yolov11nCCSDB提前停止103轮增加P2 总参数量:", total_params)

print()
# 逐层统计
'''for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} | 形状: {param.shape} | 数据类型: {param.dtype}")'''