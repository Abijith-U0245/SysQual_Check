from ultralytics import YOLO
import cv2

# ==========================
# Load trained model
# ==========================

model = YOLO("runs/train_yolo11n_2/weights/best.pt")
# Change path if your weights are elsewhere

# ==========================
# Open Camera
# ==========================

cap = cv2.VideoCapture(1)

# For USB camera use 1 if needed
# cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press Q to Quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------
    # Run YOLO
    # --------------------------

    results = model.predict(
        frame,
        imgsz=640,
        conf=0.50,
        verbose=False
    )

    annotated = results[0].plot()

    # --------------------------
    # Display detected classes
    # --------------------------

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            name = model.names[cls]

            print(f"{name} : {conf:.2f}")

    cv2.imshow("WG Tech Inspection", annotated)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()