import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

attendance_file = "attendance.csv"

try:
    attendance = pd.read_csv(attendance_file)
except:
    attendance = pd.DataFrame(columns=["Name", "Date", "Time"])

marked_names = set()
camera = cv2.VideoCapture(0)

def generate_frames():
    global attendance

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = data["names"][match_index]

                if name not in marked_names:
                    now = datetime.now()
                    attendance.loc[len(attendance)] = [
                        name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    ]
                    marked_names.add(name)
                    attendance.to_csv(attendance_file, index=False)

            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')