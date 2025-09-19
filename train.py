from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(data='d:/VScode/ultralytics-8.3.202/ultralytics-8.3.202/icon.yaml', workers=0, epochs=300, batch=16)