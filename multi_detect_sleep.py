"""
============================================================
  MULTI STUDENT SLEEP DETECTION SYSTEM
  File: multi_detect_sleep.py

  - Detects MULTIPLE students at the same time
  - Each student tracked independently
  - If ANY student sleeps 10 seconds → email sent
  - Alarm rings automatically

  Run:
    python multi_detect_sleep.py
============================================================
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import os
import smtplib
import pygame
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ══════════════════════════════════════════════
#  ✏️  FILL THESE IN
# ══════════════════════════════════════════════
EMAIL_SENDER      = "youremail@gmail.com"       # Your Gmail
EMAIL_PASSWORD    = "xxxx xxxx xxxx xxxx"        # Gmail App Password
RECEIVER_EMAIL    = "receiver@gmail.com"         # Email to receive alert

SLEEP_THRESHOLD_SEC = 10
SCREENSHOT_DIR      = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════
print("Loading model...")
model        = tf.keras.models.load_model("eye_state_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
print("Model loaded!\n")

# ══════════════════════════════════════════════
#  ALARM
# ══════════════════════════════════════════════
pygame.mixer.init()

def play_alarm():
    try:
        pygame.mixer.music.load("alarm.wav")
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Alarm error: {e}")

def stop_alarm():
    try:
        pygame.mixer.music.stop()
    except:
        pass

# ══════════════════════════════════════════════
#  EMAIL FUNCTION
# ══════════════════════════════════════════════
def send_email(student_id, elapsed, screenshot_path):
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"ALERT: Student {student_id} Is Sleeping!"
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = RECEIVER_EMAIL

        html = f"""
        <html>
        <body style="font-family:Arial,sans-serif; background:#f0f0f0; padding:20px;">
          <div style="max-width:520px; margin:auto; background:white;
                      border-radius:12px; overflow:hidden;
                      box-shadow:0 2px 10px rgba(0,0,0,0.15);">

            <div style="background:#c0392b; padding:25px 30px;">
              <h2 style="color:white; margin:0;">⚠️ Sleeping Student Detected</h2>
              <p style="color:#f5b7b1; margin:6px 0 0; font-size:14px;">
                Multi Student Sleep Detection System
              </p>
            </div>

            <div style="padding:30px;">
              <p style="color:#333;">A student has been detected sleeping!</p>

              <table style="width:100%; border-collapse:collapse; font-size:14px;">
                <tr style="background:#fdf2f2;">
                  <td style="padding:12px; font-weight:bold; color:#c0392b;">Student</td>
                  <td style="padding:12px; color:#333;">Student {student_id}</td>
                </tr>
                <tr>
                  <td style="padding:12px; font-weight:bold; color:#c0392b;">Detected At</td>
                  <td style="padding:12px; color:#333;">{now}</td>
                </tr>
                <tr style="background:#fdf2f2;">
                  <td style="padding:12px; font-weight:bold; color:#c0392b;">Eyes Closed For</td>
                  <td style="padding:12px; color:#c0392b; font-weight:bold;">{elapsed:.1f} seconds</td>
                </tr>
              </table>

              <p style="color:#555; font-size:13px; margin-top:20px;">
                Screenshot is attached for your reference.
                The alarm buzzer has been triggered automatically.
              </p>
            </div>
          </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html, "html"))

        if screenshot_path and os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                img = MIMEImage(f.read(), name=os.path.basename(screenshot_path))
                msg.attach(img)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, RECEIVER_EMAIL, msg.as_string())

        print(f"Email sent → Student {student_id} sleeping ({elapsed:.1f}s)")

    except Exception as e:
        print(f"Email error: {e}")

# ══════════════════════════════════════════════
#  EYE PREPROCESSING
# ══════════════════════════════════════════════
def preprocess_eye(eye_img):
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = eye_img / 255.0
    return np.expand_dims(eye_img, axis=[0, -1])

# ══════════════════════════════════════════════
#  STUDENT TRACKER
#  Each face gets its own independent timer
# ══════════════════════════════════════════════
# student_states = {
#   "student_1": {
#       "closed_start": None or timestamp,
#       "alarm_sent":   True/False,
#       "email_sent":   True/False
#   }, ...
# }
student_states = {}
alarm_playing  = False

def get_student_id(fx, frame_width):
    """Assign student ID based on face horizontal position on screen"""
    zone = int((fx / frame_width) * 6) + 1   # divides screen into 6 zones
    return f"Student-{zone}"

# ══════════════════════════════════════════════
#  START WEBCAM
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

print("Multi Student Sleep Detection Running...")
print("Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                          minNeighbors=5, minSize=(60, 60))

    # Header bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Multi Student Sleep Detection  |  Students Detected: {len(faces)}  |  {datetime.now().strftime('%H:%M:%S')}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    detected_ids = set()
    any_sleeping = False

    for (fx, fy, fw, fh) in faces:

        student_id = get_student_id(fx, frame_width)
        detected_ids.add(student_id)

        roi_gray  = gray[fy:fy+fh, fx:fx+fw]
        roi_color = frame[fy:fy+fh, fx:fx+fw]

        eyes      = eye_cascade.detectMultiScale(roi_gray, 1.1, 8)
        eyes_open = False

        for (ex, ey, ew, eh) in eyes:
            eye_img    = roi_gray[ey:ey+eh, ex:ex+ew]
            processed  = preprocess_eye(eye_img)
            prediction = model.predict(processed, verbose=0)[0][0]

            if prediction > 0.5:
                label     = "OPEN"
                color     = (0, 220, 0)
                eyes_open = True
            else:
                label = "CLOSED"
                color = (0, 0, 255)

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(roi_color, label, (ex, ey - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Get or create state for this student
        state = student_states.get(student_id, {
            "closed_start": None,
            "alarm_sent":   False,
            "email_sent":   False
        })

        # ── Per Student Sleep Logic ────────────────
        if not eyes_open:
            if state["closed_start"] is None:
                state["closed_start"] = time.time()

            elapsed = time.time() - state["closed_start"]

            # Red face box
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 3)
            cv2.putText(frame, f"{student_id}: {elapsed:.1f}s",
                        (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if elapsed >= SLEEP_THRESHOLD_SEC:
                any_sleeping = True

                # Alarm — ring once for all
                if not alarm_playing:
                    alarm_playing = True
                    threading.Thread(target=play_alarm, daemon=True).start()

                # Email — send once per student per sleep event
                if not state["email_sent"]:
                    state["email_sent"] = True

                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"{SCREENSHOT_DIR}/sleep_{student_id}_{ts}.jpg"
                    cv2.imwrite(path, frame)

                    threading.Thread(
                        target=send_email,
                        args=(student_id, elapsed, path),
                        daemon=True
                    ).start()

        else:
            # Student is awake — reset their state
            state["closed_start"] = None
            state["email_sent"]   = False
            state["alarm_sent"]   = False

            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 220, 0), 2)
            cv2.putText(frame, f"{student_id}: Awake",
                        (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        student_states[student_id] = state

    # Remove students no longer on screen
    for sid in list(student_states.keys()):
        if sid not in detected_ids:
            student_states.pop(sid, None)

    # Stop alarm if no one is sleeping
    if not any_sleeping and alarm_playing:
        stop_alarm()
        alarm_playing = False

    # Sleeping alert banner
    if any_sleeping:
        cv2.rectangle(frame, (0, 42), (frame.shape[1], 82), (0, 0, 180), -1)
        cv2.putText(frame, "  !! SLEEPING STUDENT DETECTED !!  ALARM ON  |  EMAIL SENT",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Multi Student Sleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────
cap.release()
cv2.destroyAllWindows()
stop_alarm()
print("System stopped.")
