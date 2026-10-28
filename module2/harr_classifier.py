
import cv2
import numpy as np

# Load pretrained Haar Cascade classifiers
eye_cascade = cv2.CascadeClassifier(
    r"C:\Users\hp\Music\mini_project_gitrepo-main\module2\haarcascade_eye.xml"
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

smile_cascade = cv2.CascadeClassifier(
    r"C:\Users\hp\Music\mini_project_gitrepo-main\module2\haarcascade_smile.xml"
)

# Check whether classifiers loaded successfully
if face_cascade.empty():
    raise FileNotFoundError("Human face cascade XML file not found.")

if eye_cascade.empty():
    raise FileNotFoundError("Eye cascade XML file not found.")


# Detect faces
def adjusted_detect_face(img):
    face_img = img.copy()

    face_rec = face_cascade.detectMultiScale(
        img, scaleFactor=1.5, minNeighbors=2
    )

    for (x, y, w, h) in face_rec:
        cv2.rectangle(
            face_img, (x, y), (x + w, y + h), (255, 255, 255), 2
        )

    return face_img


# Detect eyes
def adjusted_detect_eye(img):
    eye_img = img.copy()

    eye_rec = eye_cascade.detectMultiScale(
        img, scaleFactor=2, minNeighbors=2
    )

    for (x, y, w, h) in eye_rec:
        cv2.rectangle(
            eye_img, (x, y), (x + w, y + h), (0, 255, 0), 2
        )

    return eye_img


# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No image found.")
        break

    flipped_img = cv2.flip(frame, 1)

    gray_img = cv2.cvtColor(
        flipped_img, cv2.COLOR_BGR2GRAY
    )

    cv2.imshow("Face Detection", adjusted_detect_face(gray_img))
    cv2.imshow("Eye Detection", adjusted_detect_eye(gray_img))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Program is being terminated.")
        break

cap.release()
cv2.destroyAllWindows()
