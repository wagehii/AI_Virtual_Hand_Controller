import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from datetime import datetime

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- Webcam & Screen Setup ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam. Please check your camera connection.")
    print("[ERROR] Exiting program.")
    exit(1)

cam_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Webcam resolution: {cam_width}x{cam_height}")

screen_width, screen_height = pyautogui.size()
print(f"[INFO] Screen resolution: {screen_width}x{screen_height}")

# --- Smoothing ---
SMOOTHING_ALPHA = 0.75
smoothed_x = screen_width  // 2
smoothed_y = screen_height // 2

# --- Gesture Thresholds & Cooldowns ---
CLICK_DISTANCE_THRESHOLD = 40
last_click_time = 0
CLICK_COOLDOWN = 0.3

VOLUME_DISTANCE_THRESHOLD = 80
last_volume_time = 0
VOLUME_COOLDOWN = 0.15

SCROLL_THRESHOLD = 3
last_palm_y = None
SCROLL_COOLDOWN = 0.005
last_scroll_time = 0

last_screenshot_time = 0
SCREENSHOT_COOLDOWN = 4.0

# --- FPS ---
fps_start_time = time.time()
fps_frame_count = 0
fps_display = 0


def get_finger_states(landmarks, handedness="Right"):
    fingers = []

    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]

    if handedness == "Right":
        fingers.append(thumb_tip.x < thumb_ip.x)
    else:
        fingers.append(thumb_tip.x > thumb_ip.x)

    finger_tip_ids = [8, 12, 16, 20]
    finger_pip_ids = [6, 10, 14, 18]

    for tip_id, pip_id in zip(finger_tip_ids, finger_pip_ids):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        fingers.append(tip.y < pip.y)

    return fingers


def euclidean_distance(lm1, lm2, frame_w, frame_h):
    x1, y1 = int(lm1.x * frame_w), int(lm1.y * frame_h)
    x2, y2 = int(lm2.x * frame_w), int(lm2.y * frame_h)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ==============================================================================
# MAIN LOOP
# ==============================================================================
print("[INFO] Virtual Hand Controller started. Press 'Q' to quit.")

while True:
    # --- Capture & Flip Frame ---
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    fps_frame_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps_display = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_start_time = time.time()

    # --- Hand Detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    gesture_label = "No Hand Detected"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = results.multi_handedness[0].classification[0].label
        lm = hand_landmarks.landmark

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        fingers_up = get_finger_states(lm, handedness_label)
        thumb, index, middle, ring, pinky = fingers_up
        fingers_up_count = sum(fingers_up)

        index_tip  = lm[8]
        thumb_tip  = lm[4]
        middle_tip = lm[12]
        wrist      = lm[0]

        # --- Gesture: Move Mouse ---
        if index and not middle and not ring and not pinky:
            gesture_label = "Mode: Moving Mouse"

            raw_x = int(index_tip.x * frame_w)
            raw_y = int(index_tip.y * frame_h)

            margin_x = 0.05
            margin_y = 0.20
            mapped_x = np.interp(raw_x,
                                 [int(frame_w * margin_x), int(frame_w * (1 - margin_x))],
                                 [0, screen_width])
            mapped_y = np.interp(raw_y,
                                 [int(frame_h * margin_y), int(frame_h * (1 - margin_y))],
                                 [0, screen_height])

            smoothed_x = SMOOTHING_ALPHA * mapped_x + (1 - SMOOTHING_ALPHA) * smoothed_x
            smoothed_y = SMOOTHING_ALPHA * mapped_y + (1 - SMOOTHING_ALPHA) * smoothed_y

            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

            cv2.circle(frame,
                       (raw_x, raw_y), 12, (0, 255, 255), cv2.FILLED)

        elif index and not middle and not ring and not pinky:
            pass

        # --- Gesture: Click (Pinch) ---
        if index and not middle and not ring and not pinky:
            click_dist = euclidean_distance(thumb_tip, index_tip, frame_w, frame_h)

            if click_dist < CLICK_DISTANCE_THRESHOLD:
                gesture_label = "Mode: Clicking!"
                current_time = time.time()

                if (current_time - last_click_time) > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = current_time
                    print("[ACTION] Left Click!")

                t_px = (int(thumb_tip.x * frame_w),  int(thumb_tip.y * frame_h))
                i_px = (int(index_tip.x * frame_w),  int(index_tip.y * frame_h))
                cv2.line(frame, t_px, i_px, (0, 0, 255), 3)
                cv2.circle(frame, t_px, 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, i_px, 10, (0, 0, 255), cv2.FILLED)

        # --- Gesture: Volume Control ---
        elif index and middle and not ring and not pinky:
            gesture_label = "Mode: Volume Control"

            vol_dist = euclidean_distance(index_tip, middle_tip, frame_w, frame_h)
            current_time = time.time()

            i_px = (int(index_tip.x  * frame_w), int(index_tip.y  * frame_h))
            m_px = (int(middle_tip.x * frame_w), int(middle_tip.y * frame_h))
            cv2.line(frame, i_px, m_px, (255, 165, 0), 3)
            cv2.circle(frame, i_px, 10, (255, 165, 0), cv2.FILLED)
            cv2.circle(frame, m_px, 10, (255, 165, 0), cv2.FILLED)

            mid_point = ((i_px[0] + m_px[0]) // 2, (i_px[1] + m_px[1]) // 2)
            cv2.putText(frame, f"Dist: {int(vol_dist)}",
                        mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

            if (current_time - last_volume_time) > VOLUME_COOLDOWN:
                if vol_dist > VOLUME_DISTANCE_THRESHOLD:
                    gesture_label = "Mode: Volume UP  ▲"
                    pyautogui.press('volumeup')
                    last_volume_time = current_time
                    print("[ACTION] Volume UP")
                else:
                    gesture_label = "Mode: Volume DOWN ▼"
                    pyautogui.press('volumedown')
                    last_volume_time = current_time
                    print("[ACTION] Volume DOWN")

        # --- Gesture: Scroll (Open Palm) ---
        elif fingers_up_count == 5:
            gesture_label = "Mode: Scrolling"

            current_palm_y = wrist.y * frame_h

            if last_palm_y is not None:
                delta_y = current_palm_y - last_palm_y
                current_time = time.time()

                if abs(delta_y) > SCROLL_THRESHOLD:
                    if (current_time - last_scroll_time) > SCROLL_COOLDOWN:
                        if delta_y < 0:
                            scroll_amount = int(abs(delta_y))
                            pyautogui.scroll(max(1, scroll_amount))
                            gesture_label = "Mode: Scrolling UP ↑"
                            print(f"[ACTION] Scroll UP ({scroll_amount})")
                        else:
                            scroll_amount = int(abs(delta_y))
                            pyautogui.scroll(-max(1, scroll_amount))
                            gesture_label = "Mode: Scrolling DOWN ↓"
                            print(f"[ACTION] Scroll DOWN ({scroll_amount})")
                        last_scroll_time = current_time

            last_palm_y = current_palm_y

        # --- Gesture: Screenshot (Fist) ---
        elif fingers_up_count == 0 or (fingers_up_count == 1 and thumb and not index):
            gesture_label = "Mode: Fist (Screenshot)"
            current_time = time.time()

            if (current_time - last_screenshot_time) > SCREENSHOT_COOLDOWN:
                timestamp = datetime.now().strftime("%Y_%m_%d___%H_%M_%S")
                filename = f"screenshot_{timestamp}.png"

                screenshot = pyautogui.screenshot()
                screenshot.save(filename)
                last_screenshot_time = current_time
                print(f"[ACTION] Screenshot saved: {filename}")
                gesture_label = f"SCREENSHOT SAVED!"

            cv2.putText(frame, "[ CAPTURING ]",
                        (frame_w // 2 - 100, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        else:
            last_palm_y = None
            gesture_label = "Gesture: Unknown"

    else:
        last_palm_y = None

    # --- UI Overlay ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame,
                gesture_label,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA)

    fps_text = f"FPS: {fps_display:.1f}"
    fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame,
                fps_text,
                (frame_w - fps_text_size[0] - 10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA)
    


    instructions = [
        "Index Only: Move Mouse",
        "Pinch: Click",
        "Index+Middle: Volume",
        "Open Palm: Scroll",
        "Fist: Screenshot | Q: Quit"
    ]
    for i, instruction in enumerate(instructions):
        cv2.putText(frame,
                    instruction,
                    (10, frame_h - 10 - (len(instructions) - 1 - i) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA)

    cv2.imshow("AI Virtual Hand Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] 'Q' pressed — exiting.")
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("[INFO] Virtual Hand Controller shut down cleanly.")