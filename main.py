import tkinter as tk
import cv2
from tkinter import ttk

class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MOTION DETECTION")
        self.root.geometry("400x150")
        
        # UI Elements
        self.start_button = ttk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(pady=10)
        
        self.stop_button = ttk.Button(root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        
        self.status_label = ttk.Label(root, text="Status: Not Running")
        self.status_label.pack(pady=10) # Fixed repeated pack and missing variable
        
        # Variables
        self.running = False
        self.cap = None
        self.prev_gray = None

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Status: Error opening camera")
            return
            
        self.running = True
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_label.configure(text="Status: Running")
        
        self.prev_gray = None
        self.detect_motion() # Starts the non-blocking loop

    def stop_detection(self):
        self.running = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_label.configure(text="Status: Not Running")
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def detect_motion(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        # Image processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialize previous frame on the very first run
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            # Schedule the next frame check without blocking
            self.root.after(10, self.detect_motion)
            return

        # Delta and threshold calculations
        delta_frame = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Contour extraction
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("MOTION DETECTION", frame)
        cv2.waitKey(1) # Keeps the OpenCV window updated

        self.prev_gray = gray.copy()
        
        # Tkinter Non-blocking loop: Schedule next execution in 10 milliseconds
        self.root.after(10, self.detect_motion)

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()
# ./motiondetc/bin/python main.py. 
#use the above to run the code
