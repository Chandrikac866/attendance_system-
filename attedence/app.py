from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import time

app = Flask(__name__)

# Camera
camera = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Attendance variables
attendance_active = True
attendance_marked = False
start_time = None

# Database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Save attendance to DB
def save_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_ = now.strftime("%H:%M:%S")
    c.execute("INSERT INTO attendance (date, time, status) VALUES (?, ?, ?)",
              (date, time_, "PRESENT"))
    conn.commit()
    conn.close()

# Camera frame generator
def generate_frames():
    global attendance_active, attendance_marked, start_time

    while attendance_active:
        success, frame = camera.read()
        if not success:
            break

        # Object detection
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                count += 1

        # Attendance logic (≥5 sec)
        if count > 0:
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time
            if elapsed >= 5 and not attendance_marked:
                attendance_marked = True
                save_attendance()
                print("Attendance Saved to Database")
        else:
            start_time = None

        # Show status on frame
        status_text = "PRESENT" if attendance_marked else "NOT MARKED"
        cv2.putText(frame, f'Attendance: {status_text}', (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # If session ends, show “Attendance Ended”
    blank_frame = 255 * np.ones((480, 640, 3), np.uint8)
    cv2.putText(blank_frame, "Attendance Ended", (100,240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    ret, buffer = cv2.imencode('.jpg', blank_frame)
    frame = buffer.tobytes()
    while True:
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def view_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance")
    records = c.fetchall()
    conn.close()
    session_status = "ACTIVE" if attendance_active else "ENDED"
    return render_template('attendance.html', records=records, session_status=session_status)

@app.route('/end_attendance')
def end_attendance():
    global attendance_active
    attendance_active = False
    print("Attendance session ended by user")
    return redirect(url_for('view_attendance'))

if __name__ == '__main__':
    app.run(debug=True)