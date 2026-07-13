import cv2
import os
import time

# =====================================================================
# DATASET CONFIGURATION
# Change this variable to the defect you are currently capturing!
# Examples: "OK_Parts", "NG_Rust", "NG_Thread", "NG_Dent"
# =====================================================================
CURRENT_CLASS = "White Patches"  # <-- CHANGE THIS BEFORE RUNNING

# Create the folder dynamically based on the class name
base_folder = "dataset"
save_folder = os.path.join(base_folder, CURRENT_CLASS)
os.makedirs(save_folder, exist_ok=True)

# =====================================================================
# CAMERA INITIALIZATION (Using your exact working setup)
# =====================================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

# Force True 1080p Resolution for the high-quality AI images
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Turn off software autofocus just in case
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

print("========================================")
print(f"SYSMAC - DATASET COLLECTOR: {CURRENT_CLASS}")
print(f"Images will be saved to: {save_folder}")
print("-> Press 's' to SNAP and save an image.")
print("-> Press 'q' to quit.")
print("========================================")

img_count = 0
flash_timer = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        break

    # Save a clean, full-resolution 1080p copy for the actual YOLO dataset
    clean_frame = frame.copy()

    # Resize strictly for the laptop display so it fits nicely on your screen
    display_frame = cv2.resize(frame, (1280, 720))

    # ==========================================
    # HEADS UP DISPLAY (HUD)
    # ==========================================
    cv2.putText(display_frame, f"CURRENT CLASS: {CURRENT_CLASS}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(display_frame, f"Images Saved: {img_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press 's' to Save | Press 'q' to Quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Big green flash effect when you press save
    if flash_timer > 0:
        cv2.putText(display_frame, "SAVED!", (540, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
        flash_timer -= 1

    cv2.imshow("ELP Dataset Collector", display_frame)

    # ==========================================
    # KEYBOARD CONTROLS
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # Creates a unique filename using a timestamp so files never overwrite
        filename = os.path.join(save_folder, f"{CURRENT_CLASS}_{int(time.time())}.jpg")
        cv2.imwrite(filename, clean_frame)
        
        img_count += 1
        flash_timer = 15  # Show the green "SAVED!" text for 15 frames
        print(f"[{img_count}] Saved -> {filename}")
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()