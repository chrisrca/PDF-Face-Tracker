import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF
import cv2
import dlib
import numpy as np
import time
import os

# Load Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to open a dialog and return the selected PDF file path
def select_pdf_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select a PDF file to open", filetypes=[("PDF files", "*.pdf")])
    return file_path

# Function to display a PDF page
def display_pdf_page(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    if page_number < 0 or page_number >= len(doc):  # Check if page_number is out of bounds
        return  # Do nothing if out of bounds
    page = doc.load_page(page_number)
    pix = page.get_pixmap()
    img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
    
    # Extract the file name from the pdf_path and use it as the window title
    pdf_name = os.path.basename(pdf_path)
    window_title = f"Song: {pdf_name}"  # Use the PDF name in the window title
    
    cv2.imshow(window_title, img)

def head_tilt_direction(landmarks):
    left_eye = landmarks.part(36)
    right_eye = landmarks.part(45)
    eye_diff = right_eye.y - left_eye.y
    if eye_diff > 25:
        return "left"
    elif eye_diff < -25:
        return "right"
    else:
        return "forward"

# Main function to handle head tilt and page flipping
def main():
    pdf_path = select_pdf_file()
    if not pdf_path:
        print("No PDF file selected.")
        return
    
    cap = cv2.VideoCapture(0)
    current_page = 0  # Initialize current_page
    last_action_time = 0
    action_cooldown = 1  # Cooldown in seconds
    doc = fitz.open(pdf_path)  # Load the PDF document to check the total number of pages

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            tilt_direction = head_tilt_direction(landmarks)
            current_time = time.time()

            if (current_time - last_action_time) > action_cooldown:
                if tilt_direction == "left":
                    current_page = max(0, current_page - 1)  # Ensure current_page doesn't go below 0
                    last_action_time = current_time
                elif tilt_direction == "right":
                    current_page = min(len(doc) - 1, current_page + 1)  # Ensure current_page doesn't exceed the max index
                    last_action_time = current_time

        display_pdf_page(pdf_path, current_page)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
