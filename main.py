import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# -------------------------
# STEP 1: Prepare Dataset
# -------------------------
dataset_path = 'dataset'
trainer_path = 'trainer'
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

faces = []
labels = []
label_dict = {}
label_id = 0

for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    if not os.path.isdir(student_folder):
        continue
    label_dict[label_id] = student_name
    for image_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(label_id)
    label_id += 1

# -------------------------
# STEP 2: Train Recognizer
# -------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save(os.path.join(trainer_path, 'trainer.yml'))
print("Training complete!")

# -------------------------
# STEP 3: Start Attendance
# -------------------------
attendance = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in detected_faces:
        face_img = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face_img)
        print("ID:", id_, "Confidence:", conf) 
        if conf < 60:  # confidence threshold
            name = label_dict[id_]
            if name not in attendance:  # mark only once per session
                attendance[name] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('Student Attendance', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------
# STEP 4: Save Attendance CSV
# -------------------------
if attendance:
    df = pd.DataFrame(list(attendance.items()), columns=['Name', 'Time'])
    df.to_csv('attendance.csv', index=False)
    print("Attendance saved to attendance.csv")
else:
    print("No attendance recorded.")
