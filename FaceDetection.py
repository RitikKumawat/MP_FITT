import cv2
import dlib
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# Load face detector
detector = dlib.get_frontal_face_detector()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create GUI window
root = Tk()
root.title("Face Detection")
root.geometry("800x600")

label = Label(root, text="Face Detection Using Dlib", font=("Helvetica", 26))
label.pack(pady=10)
# Create canvas to display video frames
canvas = Canvas(root, width=800, height=600)
canvas.pack()

def detect_faces():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert frame to ImageTk format and display on canvas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.img_tk = img_tk  # Save reference to prevent garbage collection
    canvas.create_image(0, 0, anchor=NW, image=img_tk)

    # Schedule next frame processing
    canvas.after(10, detect_faces)

# Start face detection
detect_faces()

# Run GUI main loop
root.mainloop()

# Release video capture and close GUI window
cap.release()
cv2.destroyAllWindows()