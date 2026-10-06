import sys
from ultralytics import YOLO

image = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

model = YOLO("models/best.tflite", task="detect")
print("\nClasses this model detects:", model.names)

# First run is slow (warm-up), so run twice and time the second
model.predict(image, conf=0.4, verbose=False)
results = model.predict(image, conf=0.4, verbose=False)
r = results[0]

print(f"Inference speed: {r.speed['inference']:.1f} ms")
print(f"Detections found: {len(r.boxes)}")
for box in r.boxes:
    name = model.names[int(box.cls)]
    print(f"  - {name}  ({float(box.conf):.0%} confidence)")

r.save("test_output.jpg")
print("\nSaved annotated image: test_output.jpg")