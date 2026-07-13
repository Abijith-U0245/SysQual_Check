import cv2
import os
import time

# =====================================================================
# SETUP FOLDERS
# =====================================================================
# This will automatically create folders where your Python file is saved
base_folder = "dataset"
ok_folder = os.path.join(base_folder, "OK_Parts")
ng_folder = os.path.join(base_folder, "NG_Defects")

os.makedirs(ok_folder, exist_ok=True)
os.makedirs(ng_folder, exist_ok=True)

# =====================================================================
# CAMERA INITIALIZATION (ELP 2MP GLOBAL SHUTTER)
# =====================================================================
print("Connecting to camera (This should be instant)...")

# We MUST use cv2.CAP_DSHOW to prevent the 30-second Windows lag.
# We will automatically try index 1 first (usually the USB cam), then fallback to 0.
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("Camera not found on index 1. Trying index 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("\n[CRITICAL ERROR] Windows is blocking the camera!")
    print("1. CLOSE THE WINDOWS CAMERA APP (It locks the USB port).")
    print("2. Unplug the camera USB and plug it back in.")
    print("3. Run this script again.\n")
    exit()

# Force True 1080p Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Turn off any sneaky software autofocus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

print("========================================")
print("SYSMAC - PRO DATASET COLLECTOR")
print("Folders created! Images will save to the 'dataset' folder.")
print("-> Press 's' to save an OK (Good) part.")
print("-> Press 'd' to save a DEFECT (Bad) part.")
print("-> Press 'q' to quit.")
print("========================================")

ok_count = 0
ng_count = 0
flash_timer = 0
flash_text = ""
flash_color = (0,0,0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from ELP camera.")
        break

    # Keep a copy of the pure, uncompressed 1080p frame for saving
    save_frame = frame.copy()

    # Create a smaller copy just for displaying on your laptop screen
    display_frame = cv2.resize(frame, (1280, 720))

    # ==========================================
    # USER INTERFACE (HUD)
    # ==========================================
    cv2.putText(display_frame, f"OK Count: {ok_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, f"DEFECT Count: {ng_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, "Press 's' for OK | Press 'd' for DEFECT", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Flash confirmation text on screen when a picture is taken
    if flash_timer > 0:
        cv2.putText(display_frame, flash_text, (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, flash_color, 4)
        flash_timer -= 1

    cv2.imshow("ELP Dataset Collector", display_frame)

    # ==========================================
    # KEYBOARD CONTROLS
    # ==========================================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save Good Part
        filename = os.path.join(ok_folder, f"OK_{int(time.time())}.jpg")
        cv2.imwrite(filename, save_frame)
        ok_count += 1
        flash_text = "SAVED: OK"
        flash_color = (0, 255, 0)
        flash_timer = 15 # Show flash for 15 frames
        print(f"Saved -> {filename}")

    elif key == ord('d'):
        # Save Defective Part
        filename = os.path.join(ng_folder, f"DEFECT_{int(time.time())}.jpg")
        cv2.imwrite(filename, save_frame)
        ng_count += 1
        flash_text = "SAVED: DEFECT"
        flash_color = (0, 0, 255)
        flash_timer = 15
        print(f"Saved -> {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()