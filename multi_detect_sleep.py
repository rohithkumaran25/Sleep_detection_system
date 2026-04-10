


import cv2
import numpy as np
import time
import threading
import os
import smtplib
import pygame
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# New mediapipe import style (works in all versions)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# ══════════════════════════════════════════════
#  ✏️  FILL THESE IN
# ══════════════════════════════════════════════
EMAIL_SENDER    = "youremail@gmail.com"
EMAIL_PASSWORD  = "xxxx xxxx xxxx xxxx"
RECEIVER_EMAIL  = "receiver@gmail.com"

SLEEP_THRESHOLD = 10
EAR_THRESHOLD   = 0.20
SCREENSHOT_DIR  = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
#  MEDIAPIPE FACE LANDMARKER (NEW API)
# ══════════════════════════════════════════════
# Download model file automatically
import urllib.request

MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded!\n")

base_options   = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options        = vision.FaceLandmarkerOptions(
    base_options           = base_options,
    num_faces              = 10,
    min_face_detection_confidence = 0.5,
    min_face_presence_confidence  = 0.5,
    min_tracking_confidence       = 0.5
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# Eye landmark indexes for mediapipe face landmarker
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
NOSE_TIP  = 1

# ══════════════════════════════════════════════
#  AUTO BRIGHTNESS FIX
# ══════════════════════════════════════════════
def fix_lighting(frame):
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# ══════════════════════════════════════════════
#  EAR CALCULATION
# ══════════════════════════════════════════════
def get_ear(landmarks, eye_idxs, w, h):
    pts = np.array([
        [int(landmarks[i].x * w), int(landmarks[i].y * h)]
        for i in eye_idxs
    ])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return ((A + B) / (2.0 * C)) if C != 0 else 0, pts

# ══════════════════════════════════════════════
#  ALARM
# ══════════════════════════════════════════════
pygame.mixer.init()
alarm_on = False

def play_alarm():
    try:
        pygame.mixer.music.load("alarm.wav")
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Alarm error: {e}")

def stop_alarm():
    global alarm_on
    try:
        pygame.mixer.music.stop()
    except:
        pass
    alarm_on = False

# ══════════════════════════════════════════════
#  EMAIL
# ══════════════════════════════════════════════
def send_email(student_id, elapsed, path):
    try:
        now            = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"ALERT: {student_id} Is Sleeping!"
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = RECEIVER_EMAIL

        html = f"""
        <html>
        <body style="font-family:Arial; background:#f0f0f0; padding:20px;">
          <div style="max-width:500px; margin:auto; background:white;
                      border-radius:12px; overflow:hidden;">
            <div style="background:#c0392b; padding:20px 25px;">
              <h2 style="color:white; margin:0;">⚠️ Sleeping Student Alert</h2>
            </div>
            <div style="padding:25px;">
              <table style="width:100%; border-collapse:collapse; font-size:14px;">
                <tr style="background:#fdf2f2;">
                  <td style="padding:10px; font-weight:bold; color:#c0392b;">Student</td>
                  <td style="padding:10px;">{student_id}</td>
                </tr>
                <tr>
                  <td style="padding:10px; font-weight:bold; color:#c0392b;">Time</td>
                  <td style="padding:10px;">{now}</td>
                </tr>
                <tr style="background:#fdf2f2;">
                  <td style="padding:10px; font-weight:bold; color:#c0392b;">Slept For</td>
                  <td style="padding:10px; color:#c0392b; font-weight:bold;">{elapsed:.1f} sec</td>
                </tr>
              </table>
              <p style="color:#777; font-size:12px; margin-top:15px;">
                Screenshot attached. Alarm triggered automatically.
              </p>
            </div>
          </div>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        if path and os.path.exists(path):
            with open(path, "rb") as f:
                msg.attach(MIMEImage(f.read(), name=os.path.basename(path)))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, RECEIVER_EMAIL, msg.as_string())

        print(f"Email sent → {student_id} slept {elapsed:.1f}s")

    except Exception as e:
        print(f"Email error: {e}")

# ══════════════════════════════════════════════
#  STUDENT TRACKER
# ══════════════════════════════════════════════
students        = {}
student_counter = 0

def match_student(cx):
    global student_counter
    for key in list(students.keys()):
        if abs(key - cx) < 100:
            data = students.pop(key)
            students[cx] = data
            return cx
    student_counter += 1
    students[cx] = {
        "id":           f"Student-{student_counter}",
        "closed_start": None,
        "email_sent":   False
    }
    return cx

# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Multi Student Sleep Detection V5 Running...")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w   = frame.shape[:2]
    fixed  = fix_lighting(frame)

    # Convert to mediapipe image
    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data         = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
    )

    result       = landmarker.detect(mp_image)
    detected_cxs = set()
    any_sleeping = False

    # Top bar
    brightness = int(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
    light_mode = "Bright" if brightness > 130 else "Normal" if brightness > 80 else "Dark"
    num_faces  = len(result.face_landmarks) if result.face_landmarks else 0
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.putText(frame,
                f"Sleep Detection V5  |  Faces: {num_faces}  |  Light: {light_mode}  |  {datetime.now().strftime('%H:%M:%S')}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    for face_lms in (result.face_landmarks or []):

        cx  = int(face_lms[NOSE_TIP].x * w)
        key = match_student(cx)
        state = students[key]
        detected_cxs.add(key)

        # Face bounding box
        xs  = [int(lm.x * w) for lm in face_lms]
        ys  = [int(lm.y * h) for lm in face_lms]
        x1  = max(min(xs) - 10, 0)
        y1  = max(min(ys) - 10, 0)
        x2  = min(max(xs) + 10, w)
        y2  = min(max(ys) + 10, h)

        # EAR
        left_ear,  left_pts  = get_ear(face_lms, LEFT_EYE,  w, h)
        right_ear, right_pts = get_ear(face_lms, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        eyes_open = avg_ear >= EAR_THRESHOLD

        # Draw
        eye_color = (0, 220, 0) if eyes_open else (0, 0, 255)
        cv2.polylines(frame, [left_pts],  True, eye_color, 1)
        cv2.polylines(frame, [right_pts], True, eye_color, 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), eye_color, 2)
        cv2.putText(frame, f"EAR:{avg_ear:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        # Sleep logic
        if not eyes_open:
            if state["closed_start"] is None:
                state["closed_start"] = time.time()
            elapsed = time.time() - state["closed_start"]
            cv2.putText(frame, f"{state['id']}: SLEEPING {elapsed:.1f}s",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if elapsed >= SLEEP_THRESHOLD:
                any_sleeping = True
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=play_alarm, daemon=True).start()
                if not state["email_sent"]:
                    state["email_sent"] = True
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"{SCREENSHOT_DIR}/{state['id']}_{ts}.jpg"
                    cv2.imwrite(path, frame)
                    threading.Thread(
                        target=send_email,
                        args=(state["id"], elapsed, path),
                        daemon=True
                    ).start()
        else:
            state["closed_start"] = None
            state["email_sent"]   = False
            cv2.putText(frame, f"{state['id']}: Awake",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        students[key] = state

    # Remove lost faces
    for k in list(students.keys()):
        if k not in detected_cxs:
            students.pop(k, None)

    if not any_sleeping and alarm_on:
        stop_alarm()

    if any_sleeping:
        cv2.rectangle(frame, (0, 40), (w, 78), (0, 0, 170), -1)
        cv2.putText(frame, "  !! SLEEPING DETECTED !!  ALARM ON  |  EMAIL SENT",
                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Multi Student Sleep Detection V5", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
print("System stopped.")