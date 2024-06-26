import ultralytics
from ultralytics import YOLO

model = YOLO('bundesliga_cv_project/models/best.pt')
results = model.predict('bundesliga_cv_project/input_video/A1606b0e6_0 (18).mp4', save=True)
print(results[0])

for boxes in results[0].boxes:
    print(boxes)

