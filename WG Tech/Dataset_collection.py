import cv2
import numpy as np

# =====================================================================
# THE Z-HEIGHT CALIBRATION DATABASE
# =====================================================================
CALIBRATION_DB = {
    "1_M22_ADAPTER": { "OD_HEX": 25.40, "ID": 25.85, "LENGTH": 23.10 },
    "2_CAPNUT": { "OD_HEX": 26.10, "ID": 26.40, "LENGTH": 24.00 },
    "3_PART_C": { "OD_HEX": 25.0, "ID": 25.0, "LENGTH": 25.0 },
    "4_PART_D": { "OD_HEX": 25.0, "ID": 25.0, "LENGTH": 25.0 }
}

# Dummy function needed for OpenCV Trackbars
def empty(a):
    pass

# =====================================================================
# INITIALIZE CAMERA & SETTINGS UI
# =====================================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Try 0 if it fails
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Force 1080p for ELP Camera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create a window specifically for our Tuning Sliders
cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Settings", 400, 250)

# Create sliders to tune the image processing live!
cv2.createTrackbar("Blur", "Settings", 7, 21, empty) # Must be odd number
cv2.createTrackbar("Canny Min", "Settings", 20, 255, empty)
cv2.createTrackbar("Canny Max", "Settings", 80, 255, empty)
cv2.createTrackbar("Dilate", "Settings", 3, 10, empty)

part_names = list(CALIBRATION_DB.keys())
active_part = part_names[0]
modes = ['CIRCLE', 'HEX', 'BOX']
current_mode_index = 0

print("========================================")
print("SYSMAC - LIVE CANNY TUNING LOADED")
print("Use the 'Settings' window to slide the values.")
print("Watch the 'Canny Edges' window to see the math happen!")
print("========================================")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize frame slightly for laptops so it fits on screen
    frame = cv2.resize(frame, (1280, 720))
    display_frame = frame.copy()

    # --- 1. GET LIVE SLIDER VALUES ---
    blur_val = cv2.getTrackbarPos("Blur", "Settings")
    if blur_val % 2 == 0: blur_val += 1 # Blur must be an odd number (1,3,5,7...)
    if blur_val < 1: blur_val = 1
    
    canny_min = cv2.getTrackbarPos("Canny Min", "Settings")
    canny_max = cv2.getTrackbarPos("Canny Max", "Settings")
    dilate_val = cv2.getTrackbarPos("Dilate", "Settings")

    # --- 2. IMAGE PROCESSING PIPELINE ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to destroy dust/micro-scratches
    blurred = cv2.GaussianBlur(gray, (blur_val, blur_val), 1)
    
    # Apply Canny Edge Detection
    edged = cv2.Canny(blurred, canny_min, canny_max)
    
    # Dilate (thicken) the lines to close gaps, then Erode to smooth them
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=dilate_val)
    edged = cv2.erode(edged, kernel, iterations=1)

    # Show the raw math to the user!
    cv2.imshow("Canny Edges (The Math)", edged)

    # --- 3. CONTOUR FINDING ---
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    if valid_contours:
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        outer_contour = valid_contours[0]
        current_mode = modes[current_mode_index]
        ratios = CALIBRATION_DB[active_part]

        # --- MEASUREMENT LOGIC ---
        if current_mode == 'CIRCLE':
            (x, y), radius_out = cv2.minEnclosingCircle(outer_contour)
            cv2.circle(display_frame, (int(x), int(y)), int(radius_out), (0, 255, 0), 2)
            cv2.putText(display_frame, f"OD: {(radius_out * 2) / ratios['OD_HEX']:.3f}mm", (int(x)-50, int(y)-int(radius_out)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if len(valid_contours) > 1:
                for i in range(1, min(4, len(valid_contours))):
                    (ix, iy), radius_in = cv2.minEnclosingCircle(valid_contours[i])
                    if radius_in < (radius_out * 0.8) and abs(x - ix) < 25 and abs(y - iy) < 25:
                        cv2.circle(display_frame, (int(ix), int(iy)), int(radius_in), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"ID: {(radius_in * 2) / ratios['ID']:.3f}mm", (int(ix)-40, int(iy)+int(radius_in)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                        break

        elif current_mode == 'HEX':
            rect = cv2.minAreaRect(outer_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(display_frame, [box], 0, (0, 165, 255), 2)
            w, h = rect[1]
            hex_width_px = min(w, h)
            cv2.putText(display_frame, f"HEX: {hex_width_px / ratios['OD_HEX']:.3f}mm", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)

        elif current_mode == 'BOX':
            x, y, w, h = cv2.boundingRect(outer_contour)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(display_frame, f"L: {w / ratios['LENGTH']:.3f}mm", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # --- 4. HUD TEXT ---
    mode_color = (0, 255, 0) if modes[current_mode_index] == 'CIRCLE' else ((0, 165, 255) if modes[current_mode_index] == 'HEX' else (0, 255, 255))
    cv2.putText(display_frame, f"PART: {active_part} (Press 1,2,3,4)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, f"MODE: {modes[current_mode_index]} (Press 'm')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    cv2.imshow("Sysmac Measurement Tool", display_frame)

    # --- 5. KEY CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'): active_part = part_names[0]
    elif key == ord('2'): active_part = part_names[1]
    elif key == ord('3'): active_part = part_names[2]
    elif key == ord('4'): active_part = part_names[3]
    elif key == ord('m'): current_mode_index = (current_mode_index + 1) % len(modes)
    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()