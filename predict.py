from ultralytics import YOLO
yolo = YOLO(model="D:\\VScode\\ultralytics-8.3.202\\ultralytics-8.3.202\\icon.pt",task="detect")
result = yolo(source="D:\\VScode\\ultralytics-8.3.202\\ultralytics-8.3.202\\predict.png",save=True,conf=0.25,show=True)